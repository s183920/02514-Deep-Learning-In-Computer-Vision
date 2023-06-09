import torch
from tools import Agent
from dataloader import datasets
from models import models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class Segmentator(Agent):
    def __init__(self, 
        model, # model to use
        name = None, # name of agent
        project = "Segmentation", # name of project
        config = {}, # config dict containing hyperparameters
        use_wandb = False, # whether to use wandb
        wandb_kwargs = {}, # kwargs for wandb
        **kwargs # kwargs for updating default config):
    ):
        super().__init__(name = name, model = model, project = project, 
                         config = config, use_wandb = use_wandb, 
                         wandb_kwargs = wandb_kwargs, **kwargs)
        
    def set_dataset(self, dataset = "PhC"):
        size = 128
        dataset = datasets.get(dataset)

        batch_size = 6
        self.trainset = dataset(train=True)
        self.train_loader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=3)
        self.testset = dataset(train=False)
        self.test_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=3)
        
        print('Loaded %d training images' % len(self.trainset))
        print('Loaded %d test images' % len(self.testset))
        
    def set_model(self, model):
        model = models.get(model)
        if model is None:
            raise ValueError(f"Model not found")
        
        self.model = model(**self.config.get("model_kwargs", {}))
        self.model.to(self.device)
        

if __name__ == "__main__":
    segmentator = Segmentator(model = "unet")
    
    
    images, labels = next(iter(segmentator.train_loader))

    for i in range(6):
        plt.subplot(2, 6, i+1)
        plt.imshow(np.swapaxes(np.swapaxes(images[i], 0, 2), 0, 1))

        plt.subplot(2, 6, i+7)
        plt.imshow(labels[i].squeeze())
    plt.savefig("test.png")