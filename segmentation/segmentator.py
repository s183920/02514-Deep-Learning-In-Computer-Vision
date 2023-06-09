import torch
import torch.nn.functional as F
from tools import Agent
from dataloader import datasets
from model import models
from loss_functions import loss_functions
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from hparams import default_config


"""
Config:
Must contain the following keys: dataset, loss, optimiser
"""

class Segmentator(Agent):
    def __init__(self, 
        model = None, # model to use
        name = None, # name of agent
        project = "Segmentation", # name of project
        use_wandb = False, # whether to use wandb
        wandb_kwargs = {}, # kwargs for wandb
        **kwargs # kwargs for updating default config):
    ):
        # replace None with defaults           
        if model is None:
            model = default_config["model"]
        
        # call super init
        super().__init__(name = name, model = model, project = project, 
                         config = default_config, use_wandb = use_wandb, 
                         wandb_kwargs = wandb_kwargs, **kwargs)
        
        # set training necessities
        self.set_optimiser()
        self.set_loss()
        
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
        
    def set_loss(self):
        self.loss_fn = loss_functions.get(self.config.get("loss"))
        
    def train_step(self, X_batch, Y_batch):
        
        X_batch = X_batch.to(self.device)
        Y_batch = Y_batch.to(self.device)

        # set parameter gradients to zero
        self.optimizer.zero_grad()

        # forward
        Y_pred = self.model(X_batch)
        loss = self.loss_fn(Y_batch, Y_pred)  # forward-pass
        loss.backward()  # backward-pass
        self.optimizer.step()  # update weights
        
        return loss
        
    def train(self):
        # extract parameters from config
        num_epochs = self.config["train_kwargs"]["num_epochs"]
        
        X_test, Y_test = next(iter(self.test_loader))

        for epoch in range(num_epochs):
            print('* Epoch %d/%d' % (epoch+1, num_epochs))

            avg_loss = 0
            self.model.train()  # train mode
            for X_batch, Y_batch in self.train_loader:
                loss = self.train_step(X_batch, Y_batch)

                # calculate metrics to show the user
                avg_loss += loss / len(self.train_loader)
                
            print(' - loss: %f' % avg_loss)

            # # show intermediate results
            # model.eval()  # testing mode
            # Y_hat = F.sigmoid(model(X_test.to(device))).detach().cpu()
            # clear_output(wait=True)
            # for k in range(6):
            #     plt.subplot(2, 6, k+1)
            #     plt.imshow(np.rollaxis(X_test[k].numpy(), 0, 3), cmap='gray')
            #     plt.title('Real')
            #     plt.axis('off')

            #     plt.subplot(2, 6, k+7)
            #     plt.imshow(Y_hat[k, 0], cmap='gray')
            #     plt.title('Output')
            #     plt.axis('off')
            # plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
            # plt.show()
            
    def predict(self, data):
        self.model.eval()  # testing mode
        Y_pred = [F.sigmoid(self.model(X_batch.to(self.device))) for X_batch, _ in data]
        return np.array(Y_pred)
        

if __name__ == "__main__":
    segmentator = Segmentator()
    segmentator.train()
    
    
    # images, labels = next(iter(segmentator.train_loader))

    # for i in range(6):
    #     plt.subplot(2, 6, i+1)
    #     plt.imshow(np.swapaxes(np.swapaxes(images[i], 0, 2), 0, 1))

    #     plt.subplot(2, 6, i+7)
    #     plt.imshow(labels[i].squeeze())
    # plt.savefig("test.png")