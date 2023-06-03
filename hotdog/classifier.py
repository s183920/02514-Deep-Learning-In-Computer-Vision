import torch
from torch.nn import functional as F
from model import models
from dataloader import HotdogDataset
from tqdm import tqdm
import os
from hparams import wandb_defaults, default_config
import wandb
from PIL import Image as PILImage

from utils import data_to_img_array

from datetime import datetime as dt

class HotdogClassifier:
    def __init__(self, name = None, model = "SimpleCNN", config = None, use_wandb = True, verbose = True, show_test_images = False, **kwargs):
        """
        Class for training and testing a hotdog classifier
        
        Parameters
        ----------
        name : str, optional
            Name of the run, by default 'run_' + current time
        config : dict, optional
            Dictionary with configuration, by default the defualt_config from hparams.py
        use_wandb : bool, optional
            Whether to use wandb, by default True
        verbose : bool, optional
            Whether to print progress, by default True
        show_test_images : bool, optional
            Whether to show test images, by default False
        **kwargs : dict
            Additional arguments to config
        """
        
        # set info
        self.name = name if name is not None else "run_" + dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.show_test_images = show_test_images
        
        # set init values
        self.config = config if config is not None else default_config  
        self.config.update(kwargs)    
        self.dev_mode = False
        self.wandb_run = None
        self.test_images = []
        
        # set model
        self.set_model(model)
        
        # set wandb
        if use_wandb:
            self.set_logger()
            
        
        # set dataset
        self.set_dataset()
        
        
        
     
    def set_model(self, model):
        model = models.get(model) 
        if model is None:
            raise ValueError(f"Model not found")
        self.model = model()
        self.model.to(self.device)
        
        
    def set_dataset(self):
        self.data_train = HotdogDataset(**self.config.get("train_dataset_kwargs", {}))
        self.data_test = HotdogDataset(train=False, **self.config.get("test_dataset_kwargs", {}))
             
        self.train_loader = self.data_train.get_dataloader(**self.config.get("train_dataloader_kwargs", {}))
        self.test_loader = self.data_test.get_dataloader(**self.config.get("test_dataloader_kwargs", {}))
    
    def set_optimizer(self):
        optimizer = self.config.get("optimizer")
        self.optimizer = torch.optim.__dict__.get(optimizer)(self.model.parameters(), **self.config.get("optimizer_kwargs", {}))
        
    def save_model(self, path, epoch):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        
        # add artifacts
        if self.wandb_run is not None:
            artifact = wandb.Artifact(self.name + "_model", type="model", description=f"model trained on hotdog dataset after {epoch} epochs")
            artifact.add_file(path)
            self.wandb_run.log_artifact(artifact)
        
    def set_logger(self, **kwargs):
        # overwrite defaults with parsed arguments
        wandb_settings = wandb_defaults.copy()
        wandb_settings.update(kwargs)
        wandb_settings.update({"dir" : "logs/" + wandb_settings.get("project") + self.name}) # create log dir
        wandb_settings.update({"name" : self.name, "group": self.model.name}) # set run name

        # create directory for logs if first run in project
        os.makedirs(wandb_settings["dir"], exist_ok=True)

        # init wandb       
        self.wandb_run = wandb.init(**wandb_settings, config = self.config)

        # setup wandb config
        self.config = self.wandb_run.config
        
        # watch model
        self.wandb_run.watch(self.model)
        
  
    def prepare_train(self, cuda_device):
        # set visible cuda devices
        if self.device.type == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device).strip("[]").replace(" ", "")
            
        # set loss function
        loss_fun = self.config.get("loss_fun")
        if loss_fun == "CrossEntropyLoss":
            self.loss_fun = lambda output, target: F.cross_entropy(output, target) 
        else:
            self.loss_fun = lambda output, target: F.nll_loss(torch.log(output), target)
            
        # set optimizer
        self.set_optimizer()
        
    
    def train_step(self, data, target):
        # send data to device
        data, target = data.to(self.device), target.to(self.device)
        
        #Zero the gradients computed for each weight
        self.optimizer.zero_grad()
        
        #Forward pass your image through the network
        output = self.model(data)
        
        #Compute the loss
        loss = self.loss_fun(output, target)
        
        #Backward pass through the network
        loss.backward()
        
        #Update the weights
        self.optimizer.step()
        
        #Compute how many were correctly classified
        predicted = output.argmax(1)
        train_correct = (target==predicted).sum().cpu().item()
        
        return train_correct, loss

            
    def train(self, num_epochs=None, cuda_device = [0]):
        # prepare training
        num_epochs = self.config.get("num_epochs") if num_epochs is None else num_epochs
        self.prepare_train(cuda_device)
        print(f"Starting training on {self.device.type}")
        
        for epoch in tqdm(range(num_epochs), unit='epoch'):
            train_correct = 0
            self.model.train()
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for minibatch_no, (data, target) in pbar:
                train_correct_, train_loss = self.train_step(data, target)
                train_correct += train_correct_
                
                # break if dev mode
                if self.dev_mode:
                    break
                
            #Comput the train accuracy
            train_acc = train_correct/len(self.data_train)*100
            if self.verbose:
                print("Accuracy train: {train:.1f}%".format(train=train_acc))
            

            # # Save model
            self.save_model(f"logs/Hotdog/models/{self.name}.pth", epoch)
            
            # test 
            test_acc, test_loss, conf_mat = self.test()
            
            
            # log to wandb
            if self.wandb_run is not None:
                self.wandb_run.log({"Validation metrics/" + key : value for key, value in conf_mat.items()}, commit = False)
                self.wandb_run.log({
                    "Train metrics/train_acc":    train_acc,
                    "Train metrics/train_loss":   train_loss,
                    "Validation metrics/test_acc":     test_acc,
                    "Validation metrics/test_loss":    test_loss,
                    "epoch":        epoch+1,
                    "test_images":  self.test_images,
                })
            
            
            
    def test(self):        
        if self.verbose:
            print("Testing model...")
        
        # Init counters
        test_correct = 0
        test_loss = 0
        
        # Test the model
        self.model.eval()
        for data, target in self.test_loader:
            data = data.to(self.device)
            with torch.no_grad():
                output = self.model(data)
            predicted = output.argmax(1).cpu()
            
            # Update counters
            test_correct += (target==predicted).sum().item()
            test_loss += self.loss_fun(output.cpu(), target).item()
            
        # compute stats        
        test_acc = test_correct/len(self.data_test)*100
        test_loss /= len(self.test_loader)
        
        # calculate confusion matrix
        true_positive, true_negative, false_positive, false_negative = self.calculate_confusion_matrix(target, predicted)
        # conf_mat = self.calculate_confusion_matrix(data, target, predicted)
        conf_mat = {"true_positive": true_positive, "true_negative": true_negative, "false_positive": false_positive, "false_negative": false_negative}
        
        
        if self.verbose:
            print("Accuracy test: {test:.1f}%".format(test=test_acc))
        
        if self.show_test_images:
            self.create_test_images(data, target, predicted, output)
        
        return test_acc, test_loss, conf_mat
    
    def calculate_confusion_matrix(self, target, predicted, idx = False):
        print(target.shape, predicted.shape)
        true_positive = ((target==predicted) & (target==1))
        true_negative = ((target==predicted) & (target==0))
        false_positive = ((target!=predicted) & (target==0))
        false_negative = ((target!=predicted) & (target==1))
        
        if idx:
           return true_positive, true_negative, false_positive, false_negative
        
        
        # calculate stats
        true_positive = true_positive.sum().item()
        true_negative = true_negative.sum().item()
        false_positive = false_positive.sum().item()
        false_negative = false_negative.sum().item()
        
        print(f"True positive: {true_positive}")
        print(f"True negative: {true_negative}")
        print(f"False positive: {false_positive}")
        print(f"False negative: {false_negative}")
 
        return true_positive, true_negative, false_positive, false_negative
            
            
    def create_test_images(self, data, target, predicted, output):
        
        # get confusion matrix
        true_positive, true_negative, false_positive, false_negative = self.calculate_confusion_matrix(target, predicted, idx = True)

        # get 3 random misclassified images
        misclassified = data_to_img_array(data[(target!=predicted)][:3].cpu())
        correct_classified = data_to_img_array(data[(target==predicted)][:3].cpu())
        
        # log images
        test_images = []
        for i in range(min(3, len(misclassified))):
            pil_image = PILImage.fromarray(misclassified[i], mode="RGB")
            image = wandb.Image(pil_image, caption=f"Misclassified {i}, pred = {torch.exp(output[:i])}")
            test_images.append(image)
            
            # pil_image = PILImage.fromarray(correct_classified[i].squeeze().transpose(1,2,0), mode="RGB")
            # image = wandb.Image(pil_image, caption=f"Correct classified {i}")
            # test_images.append(image)
        
        # wandb.log({"test_images": test_images})
        self.test_images = test_images
            
        
        
if __name__ == "__main__":
    classifier = HotdogClassifier(show_test_images=False)
    # classifier.dev_mode = True
    classifier.train(num_epochs=10)