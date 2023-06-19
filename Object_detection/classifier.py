from hparams import default_classifier_config, wandb_defaults
import torch
import torch.nn.functional as F
from datetime import datetime as dt
from utils import download_model
from model import models
import os
import wandb
from tqdm import tqdm

class TacoClassifier:
    def __init__(self, project = "Taco", name = None, model = None, config = None, use_wandb = True, verbose = True, show_test_images = False, model_save_freq = None, **kwargs):
        """
        Class for training and testing a taco classifier

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
            Additional arguments to TacoDataset for training
            i.e. train_dataset_kwargs = {"data_augmentation": False}
        test_dataset_kwargs : dict
            Additional arguments to TacoDataset for testing
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
        self.config = config if config is not None else default_classifier_config
        self.config.update(kwargs)
        self.dev_mode = False
        self.wandb_run = None
        self.test_images = []

        # set model
        # print(f"Setting model to {model}")
        self.set_model("Resnet50" if model is None else model)


        # set wandb
        if use_wandb:
            self.set_logger(project = project)


        # set dataset
        self.set_dataset()
        
    
    def load_model(self, path, model_name = "model.pth"):
        print(f"Loading model from {path}")
        if path.startswith("wandb:"):
            dl_path = "logs/Taco/models/latest_from_wandb"
            path = path[6:]
            print(f"Downloading model from wandb: {path}")
            path = download_model(path, dl_path)
        
        self.model.load_state_dict(torch.load(path+"/"+model_name, map_location=torch.device(self.device)))           
        
    def set_model(self, model):
        print(f"Setting model to {model}")
        
        model = models.get(model.lower())
        if model is None:
            raise ValueError(f"Model not found")
      
        self.model = model(finetune = self.config["finetune"])

        self.model.to(self.device)


    def set_dataset(self):
        raise NotImplementedError("set_dataset not implemented")
    #     self.data_train = HotdogDataset(**self.config.get("train_dataset_kwargs", {}))
    #     self.data_test = HotdogDataset(train=False, **self.config.get("test_dataset_kwargs", {}))
             
    #     self.train_loader, self.val_loader = self.data_train.get_dataloader(**self.config.get("train_dataloader_kwargs", {}))
    #     self.test_loader = self.data_test.get_dataloader(**self.config.get("test_dataloader_kwargs", {}))

    def set_optimizer(self):
        optimizer = self.config.get("optimizer")
        
        # set default lr if not set
        if optimizer.lower() == "sgd" and "lr" not in self.config:
            self.config["lr"] = 0.01
            
        # set optimizer
        self.optimizer = torch.optim.__dict__.get(optimizer)(self.model.parameters(), lr = self.config["lr"])
   

    def save_model(self, path, epoch):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

        # add artifacts
        if self.wandb_run is not None:
            artifact = wandb.Artifact(self.name + "_model", type="model", description=f"model trained on taco dataset after {epoch} epochs")
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
            train_loss = 0
            self.model.train()
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for minibatch_no, (data, target) in pbar:
                train_correct_, train_loss_ = self.train_step(data, target)
                train_correct += train_correct_
                train_loss += train_loss_.item()

                # break if dev mode
                if self.dev_mode:
                    break

            #Comput the train accuracy
            train_loss /= len(self.train_loader)
            train_acc = train_correct/len(self.data_train)*100
            if self.verbose:
                print("Accuracy train: {train:.1f}%".format(train=train_acc))
            
            
            # test 
            val_acc, val_loss, conf_mat = self.test(validation=True)            
            
            
            # Save model
            if self.model_save_freq is None:
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.save_model(f"logs/Taco/models/{self.name}/model.pth", epoch)
            elif epoch % self.model_save_freq == 0:
                self.save_model(f"logs/Taco/models/{self.name}/model.pth", epoch)
            
            
            # log to wandb
            if self.wandb_run is not None:
                # self.wandb_run.log({"Validation metrics/" + key : value for key, value in conf_mat.items()}, commit = False)
                self.wandb_run.log({
                    "Train metrics/train_acc":    train_acc,
                    "Train metrics/train_loss":   train_loss,
                    "Validation metrics/val_acc":     val_acc,
                    "Validation metrics/val_loss":    val_loss,
                    "epoch":        epoch,
                })
            
            # clear cache
            self.clear_cache()
            
        # log best val acc
        if self.wandb_run is not None:
            self.wandb_run.log({"Best validation accuracy": self.best_val_acc}, commit = True)
            
            
    def test(self, validation = False):               
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
    
        # compute stats
        test_acc = test_correct/data_len*100
        test_loss /= len(data_loader)

        
        
        # calculate confusion matrix      
        test_acc = test_correct/data_len*100
        test_loss /= len(data_loader)

        
        if self.verbose:
            print("Accuracy test: {test:.1f}%".format(test=test_acc))


        return test_acc, test_loss  


    def clear_cache(self):
        os.system("rm -rf ~/.cache/wandb")
        



if __name__ == "__main__":
    classifier = TacoClassifier(project="TacoClassifier", name = None, 
                                show_test_images=False, model = "resnet50", use_wandb=True, 
                                optimizer = "Adam",)

    classifier.train(num_epochs=100)

    # classifier.test(save_images=10)