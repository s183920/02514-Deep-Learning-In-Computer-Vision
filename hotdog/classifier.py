import torch
from torch.nn import functional as F
import torchvision
from model import models
from dataloader import HotdogDataset
from tqdm import tqdm
import os
from hparams import wandb_defaults, default_config, sweep_defaults
import wandb
from PIL import Image as PILImage
from torchcam.utils import overlay_mask
from torchcam.methods import SmoothGradCAMpp, LayerCAM
from torchvision.transforms.functional import normalize, resize, to_pil_image
import matplotlib.pyplot as plt
import numpy as np


from utils import data_to_img_array, download_model, ReduceLROnPlateau, inverse_normalize

from datetime import datetime as dt

class HotdogClassifier:
    def __init__(self, project = "Hotdog", name = None, model = None, config = None, use_wandb = True, verbose = True, show_test_images = False, model_save_freq = None, **kwargs):
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
        model_save_freq : int, optional
            How often to save the model, by default None
            None means only save the best model
        **kwargs : dict
            Additional arguments to config
            
        Config
        ------
        num_epochs : int
            Number of epochs to train
        dropout : float
            Dropout rate
        batchnorm : bool
            Whether to use batchnormalisation
        train_dataset_kwargs : dict
            Additional arguments to HotdogDataset for training
            i.e. train_dataset_kwargs = {"data_augmentation": False}
        test_dataset_kwargs : dict
            Additional arguments to HotdogDataset for testing
        optimizer : str
            Optimizer to use (from torch.optim)
        optimizer_kwargs : dict
            Additional arguments to optimizer
            e.g. optimizer_kwargs = {"lr": 0.01}
        scheduler : bool
            Whether to use a scheduler - will use ExponentialLR with gamma = 0.1
            Decrease will happen after 20 % of epochs
        """

        # set info
        self.name = name if name is not None else "run_" + dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.show_test_images = show_test_images
        self.model_save_freq = model_save_freq

        # set init values
        self.config = config if config is not None else default_config
        self.config.update(kwargs)
        self.dev_mode = False
        self.wandb_run = None
        self.test_images = []

        # set model
        # print(f"Setting model to {model}")
        self.set_model("SimpleCNN" if model is None else model)


        # set wandb
        if use_wandb:
            self.set_logger(project = project)


        # set dataset
        self.set_dataset()
        
    
    def load_model(self, path, model_name = "model.pth"):
        print(f"Loading model from {path}")
        if path.startswith("wandb:"):
            dl_path = "logs/Hotdog/models/latest_from_wandb"
            path = path[6:]
            print(f"Downloading model from wandb: {path}")
            path = download_model(path, dl_path)
        
        self.model.load_state_dict(torch.load(path+"/"+model_name, map_location=torch.device(self.device)))           
        
    def set_model(self, model):
        print(f"Setting model to {model}")
        
        transfer_learning = model.lower() in ["resnet18"]
        # if model.lower() == "resnet18":
        #     self.model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        #     self.model.name = "resnet18"
        # else:
        model = models.get(model)
        if model is None:
            raise ValueError(f"Model not found")
        
        if transfer_learning:
            self.model = model(dropout = self.config["dropout"], batchnorm = self.config["batchnorm"], finetune = self.config["finetune"])
        else:
            self.model = model(dropout = self.config["dropout"], batchnorm = self.config["batchnorm"])
        
        self.model.to(self.device)


    def set_dataset(self):
        self.data_train = HotdogDataset(**self.config.get("train_dataset_kwargs", {}))
        self.data_test = HotdogDataset(train=False, **self.config.get("test_dataset_kwargs", {}))
             
        self.train_loader, self.val_loader = self.data_train.get_dataloader(**self.config.get("train_dataloader_kwargs", {}))
        self.test_loader = self.data_test.get_dataloader(**self.config.get("test_dataloader_kwargs", {}))

    def set_optimizer(self):
        optimizer = self.config.get("optimizer")
        
        # set default lr if not set
        if optimizer.lower() == "sgd" and "lr" not in self.config["optimizer_kwargs"]:
            self.config["optimizer_kwargs"]["lr"] = 0.01
            
        # set optimizer
        self.optimizer = torch.optim.__dict__.get(optimizer)(self.model.parameters(), **self.config.get("optimizer_kwargs", {}))
        # self.optimizer = torch.optim.__dict__.get(optimizer)(self.model.parameters(), )
        
        
        
        if self.config.get("use_scheduler", True):
            # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=0.1, step_size = int(5+0.2*self.config["num_epochs"]))
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        else:
            self.scheduler = None

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
        wandb_settings.update({"dir" : "logs/" + wandb_settings.get("project") + "/" + self.name}) # create log dir
        wandb_settings.update({"name" : self.name, "group": self.model.name}) # set run name

        # create directory for logs if first run in project
        os.makedirs(wandb_settings["dir"], exist_ok=True)

        # init wandb
        self.wandb_run = wandb.init(**wandb_settings, config = self.config)

        # setup wandb config
        self.config = self.wandb_run.config

        # watch model
        self.wandb_run.watch(self.model)


    def prepare(self, cuda_device):
        # set seed
        if self.config.get("seed") is not None:
            torch.manual_seed(self.config.get("seed", 0))

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
        
        # set best acc
        self.best_val_acc = 0


    def train_step(self, data, target):
        # send data to device
        data, target = data.to(self.device), target.to(self.device)

        #Zero the gradients computed for each weight
        self.optimizer.zero_grad()

        #Forward pass your image through the network
        output = self.model(data).float() # kat her

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
        self.prepare(cuda_device)
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
            
            
            # test 
            val_acc, val_loss, conf_mat = self.test(validation=True)
            
            # take step in scheduler
            if self.scheduler:
                self.scheduler.step(val_loss)
                if self.wandb_run is not None:
                    self.wandb_run.log({"Learning rate" : self.scheduler.get_last_lr()[0]}, commit = False)
                
            
            
            
            # Save model
            if self.model_save_freq is None:
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.save_model(f"logs/Hotdog/models/{self.name}/model.pth", epoch)
            elif epoch % self.model_save_freq == 0:
                self.save_model(f"logs/Hotdog/models/{self.name}/model.pth", epoch)
            
            
            # log to wandb
            if self.wandb_run is not None:
                self.wandb_run.log({"Validation metrics/" + key : value for key, value in conf_mat.items()}, commit = False)
                self.wandb_run.log({
                    "Train metrics/train_acc":    train_acc,
                    "Train metrics/train_loss":   train_loss,
                    "Validation metrics/val_acc":     val_acc,
                    "Validation metrics/val_loss":    val_loss,
                    "epoch":        epoch,
                    "test_images":  self.test_images,
                })
            
            # clear cache
            self.clear_cache()
            
        # log best val acc
        if self.wandb_run is not None:
            self.wandb_run.log({"Best validation accuracy": self.best_val_acc}, commit = True)
            
            
    def test(self, validation = False, save_images = 0):  
        if save_images > 0:
            self.test_images = {"true_positive" : torch.empty(10, 3, 128, 128), "true_negative" : torch.empty(10, 3, 128, 128), "false_positive" : torch.empty(10, 3, 128, 128), "false_negative" : torch.empty(10, 3, 128, 128)}
             
        if validation:
            data_loader = self.val_loader
        else:
            self.prepare([0])
            data_loader = self.test_loader
        data_len = len(data_loader.dataset)
        
        print("Performing test with {} images".format(data_len))
        
        # Init counters
        test_correct = 0
        test_loss = 0
        
        # conf matrix
        true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
        
        # Test the model
        self.model.eval()
        for data, target in data_loader:
            data = data.to(self.device)
            with torch.no_grad():
                output = self.model(data)
            predicted = output.argmax(1).cpu()

            # Update counters
            test_correct += (target==predicted).sum().item()
            test_loss += self.loss_fun(output.cpu(), target).item()
            
            # calculate confusion matrix
            tp, tn, fp, fn = self.calculate_confusion_matrix(target, predicted)
            true_positive += tp
            true_negative += tn
            false_positive += fp
            false_negative += fn
            
            # save images
            if save_images > 0:
                self.save_images(data, target, predicted, output, save_images)

        # compute stats
        test_acc = test_correct/len(self.data_test)*100
        test_loss /= len(self.test_loader)

        
        
        # calculate confusion matrix      
        test_acc = test_correct/data_len*100
        test_loss /= len(data_loader)
        # conf_mat = {"true_positive": true_positive/data_len, "true_negative": true_negative/data_len, "false_positive": false_positive/data_len, "false_negative": false_negative/data_len}
        conf_mat = {"true_positive": true_positive, "true_negative": true_negative, "false_positive": false_positive, "false_negative": false_negative}
        
        if self.verbose:
            print("Accuracy test: {test:.1f}%".format(test=test_acc))

        if self.show_test_images:
            self.create_test_images(data, target, predicted, output)

        return test_acc, test_loss, conf_mat

    def save_images(self, data, target, predicted, output, num_images = 5):
        tp_idx, tn_idx, fp_idx, fn_idx = self.calculate_confusion_matrix(target, predicted, return_idx = True)
                
        if len(self.test_images["true_positive"]) <= num_images and sum(tp_idx) > 0:
            # self.test_images["true_positive"].extend([data[i] for i in tp_idx[:min(num_images, len(tp_idx))]])
            for i in range(min(num_images, sum(tp_idx))):
                self.test_images["true_positive"][i] = data[tp_idx][i]
        
        if len(self.test_images["true_negative"]) <= num_images and sum(tn_idx) > 0:
            # self.test_images["true_negative"].extend([data[i] for i in tn_idx[:min(num_images, len(tn_idx))]])
            for i in range(min(num_images, sum(tn_idx))):
                self.test_images["true_negative"][i] = data[tn_idx][i]
        
        if len(self.test_images["false_positive"]) <= num_images and sum(fp_idx) > 0:
            for i in range(min(num_images, sum(fp_idx))):
                self.test_images["false_positive"][i] = data[fp_idx][i]
        
        if len(self.test_images["false_negative"]) <= num_images and sum(fn_idx) > 0:
            # self.test_images["false_negative"].extend([data[i] for i in fn_idx[:min(num_images, len(fn_idx))]])
            for i in range(min(num_images, sum(fn_idx))):
                self.test_images["false_negative"][i] = data[fn_idx][i]
            

    def calculate_confusion_matrix(self, target, predicted, return_idx = False):
        # positve = hotdog (0)
        true_positive = ((target==predicted) & (target==0))
        true_negative = ((target==predicted) & (target==1))
        false_positive = ((target!=predicted) & (target==1))
        false_negative = ((target!=predicted) & (target==0))
        
        if return_idx:
           return true_positive, true_negative, false_positive, false_negative


        # calculate stats
        true_positive = true_positive.sum().item()
        true_negative = true_negative.sum().item()
        false_positive = false_positive.sum().item()
        false_negative = false_negative.sum().item()

        # print(f"True positive: {true_positive}")
        # print(f"True negative: {true_negative}")
        # print(f"False positive: {false_positive}")
        # print(f"False negative: {false_negative}")

        return true_positive, true_negative, false_positive, false_negative


    def create_test_images(self, data, target, predicted, output):

        # get confusion matrix
        true_positive, true_negative, false_positive, false_negative = self.calculate_confusion_matrix(target, predicted, idx = True)

        # get 3 random misclassified images
        misclassified = data_to_img_array(data[(target!=predicted)][:3].cpu())
        correct_classified = data_to_img_array(data[(target==predicted)][:3].cpu())

        # log images
        test_images = []
        idxs = torch.randint(0, len(target))
        # for i in range(min(3, len(misclassified))):
        for i in idxs:
            # pil_image = PILImage.fromarray(misclassified[i], mode="RGB")
            # pil_image = PILImage.fromarray(data[i].cpu(), mode="RGB")
            pil_image = data_to_img_array(data[i].cpu().unsqueeze(0))
            image = wandb.Image(pil_image, caption=f"Misclassified {i}, pred = {torch.exp(output[i]).item()}, target = {target[i].item()}")
            test_images.append(image)

            # pil_image = PILImage.fromarray(correct_classified[i].squeeze().transpose(1,2,0), mode="RGB")
            # image = wandb.Image(pil_image, caption=f"Correct classified {i}")
            # test_images.append(image)

        # wandb.log({"test_images": test_images})
        self.test_images = test_images

    def sweep(self, **kwargs):
        sweep_configuration = sweep_defaults.copy()
        sweep_configuration.update(kwargs)

        sweep_id = wandb.sweep(
            sweep=sweep_configuration,
            project='Hotdog-sweeps'
        )

        # Start sweep job.
        # wandb.agent(sweep_id, function=self.train, count=4)
        os.system(f"wandb agent {sweep_id}")

    def clear_cache(self):
        os.system("rm -rf ~/.cache/wandb")
        
    def saliency_map(self, classification_result, img_idx = 0, ax = None, layer = "convolutional"):
        assert classification_result in ["true_positive", "true_negative", "false_positive", "false_negative"]
        
        data = inverse_normalize(self.test_images[classification_result])
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            
        # Set your CAM extractor
        cam_extractor = SmoothGradCAMpp(self.model, target_layer=layer) # layer 4 is the last conv layer of the model
        # cam_extractor = SmoothGradCAMpp(self.model, target_layer='fc')
        # get your input
        input_tensor = data[img_idx].unsqueeze(0).to(self.device)

        out = self.model(input_tensor)
        # Retrieve the CAM by passing the class index and the model output
        cams = cam_extractor(out.squeeze(0).argmax().item(), out)
        
        # img = images[0]
        # img = tp_imgs[0]
        img = data_to_img_array(data)[img_idx]
        for name, cam in zip(cam_extractor.target_names, cams):
            # result = overlay_mask(to_pil_image(img, mode = "RGB"), to_pil_image(cam.squeeze(0), mode='F'), alpha=0.5)
            result = overlay_mask(PILImage.fromarray(img.astype(np.uint8)), to_pil_image(cam.squeeze(0), mode='F'), alpha=0.6)
            ax.imshow(result); ax.set_axis_off(); 
            # ax.set_title(name)
        cam_extractor.remove_hooks()


if __name__ == "__main__":
    classifier = HotdogClassifier(project="HotdogModels", name = "Resnet_finetune", show_test_images=False, model = "Resnet18", use_wandb=True, finetune =True)
    # classifier.dev_mode = True
    classifier.train(num_epochs=100)
    # classifier.sweep()


    # classifier.load_model("wandb:deepcomputer/Hotdog/Resnet18_finetune_model:v8", model_name="Resnet18_finetune.pth")
    # classifier.test(save_images=10)
    
    
    
    model = "SimpleCNN"
    trained_model = "wandb:deepcomputer/grid_search/run_2023-06-07_10-14-16_model:v8"
    classifier = HotdogClassifier(model = model, use_wandb=False)
    classifier.load_model(trained_model)
    classifier.test(save_images=2)
    # classifier.saliency_map(classifier.test_images["false_positive"][0])
    classifier.saliency_map("false_positive", img_idx = 0)
