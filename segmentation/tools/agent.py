import torch
import os
from tools.logging import download_model
from tools.optimiser import get_optimiser
import wandb
import numpy as np
from datetime import datetime as dt

class Agent(object):
    def __init__(self, 
        project, # name of project
        name, # name of agent
        model, # model to use
        config = {}, # config dict containing hyperparameters
        use_wandb = True, # whether to use wandb
        wandb_kwargs = {}, # kwargs for wandb
        **kwargs # kwargs for updating default config
    ):
        # set args
        self.config = config
        self.config.update(kwargs)
        self.project = project
        self.name = name if name is not None else "run_" + dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # set initial values
        self.wandb_run = None
        self.set_device()
        
        # set model and data set
        self.set_model(model)
        self.set_dataset()
        
        # set logger
        if use_wandb:
            self.set_logger(**wandb_kwargs)
            
    def set_dataset(self):
        raise NotImplementedError("Implement method for storing datasets and loaders for train, validation and test")
    
    def set_model(self, model):
        raise NotImplementedError("Implement method for setting model")
    
    def set_device(self, device = None, cuda_device = "0"):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        if self.device.type == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device).strip("[]").replace(" ", "")
            
        print(f"Using device: {self.device}")
        
    def clear_cache(self):
        os.system("rm -rf ~/.cache/wandb")
        
    def load_model(self, path, model_name = "model.pth"):
        print(f"Loading model from {path}")
        if path.startswith("wandb:"):
            dl_path = f"logs/{self.project}/models/latest_from_wandb"
            path = path[6:]
            print(f"Downloading model from wandb: {path}")
            path = download_model(path, dl_path)
        
        self.model.load_state_dict(torch.load(path+"/"+model_name, map_location=torch.device(self.device)))
        
    def save_model(self, path, model_name = "model.pth", description = None):
        path = os.path.join(path, model_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

        # add artifacts
        if self.wandb_run is not None:
            artifact = wandb.Artifact(self.wandb_run.id + "_model", type="model", description=description)
            artifact.add_file(path)
            self.wandb_run.log_artifact(artifact)
        
    def set_optimiser(self):
        optimizer = self.config["optimiser"]
        self.optimizer = get_optimiser(optimizer, self.model, **self.config.get("optimizer_kwargs", {}))
        
    def set_logger(self, **kwargs):
        # overwrite defaults with parsed arguments
        wandb_settings = {
            # "sync_tensorboard":True, 
            "reinit":True,
            "entity" : "deepcomputer",
            "name" : self.name,
            "project" : self.project, # wandb project name, each project correpsonds to an experiment
            # "dir" : "logs/" + "GetStarted", # dir to store the run in
            # "group" : self.agent_name, # uses the name of the agent class
            "save_code" : True,
            "mode" : "online",
        }
        wandb_settings.update(kwargs)
        wandb_settings.update({"dir" : "logs/" + wandb_settings.get("project") + "/" + self.name}) # create log dir

        # create directory for logs if first run in project
        os.makedirs(wandb_settings["dir"], exist_ok=True)

        # init wandb
        self.wandb_run = wandb.init(**wandb_settings, config = self.config)

        # setup wandb config
        self.config = self.wandb_run.config

        # watch model
        self.wandb_run.watch(self.model)
        
    def set_seed(self, seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)