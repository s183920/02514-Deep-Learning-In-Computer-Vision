import torch
import torch.nn.functional as F
from tools import Agent
from dataloader import get_datasets
from model import models
from loss_functions import loss_functions
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from hparams import default_config
import PIL
import wandb
from val_metrics import dice_overlap, IoU, accuracy, sensitivity, specificity


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
        
    def set_dataset(self):
        dataset = self.config.get("dataset")
        print(f"Loading dataset: {dataset}")
        self.trainset, self.testset, self.valset = get_datasets(dataset)

        batch_size = 6
        self.train_loader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=3)
        self.test_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=3)
        
        print('Loaded %d training images' % len(self.trainset))
        print('Loaded %d test images' % len(self.testset))
        
        if self.valset is not None:
            self.val_loader = DataLoader(self.valset, batch_size=batch_size, shuffle=False, num_workers=3)
            print('Loaded %d validation images' % len(self.valset))
        
        
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

        Y_pred = torch.nn.Sigmoid()(Y_pred)
        dice_overlap_score = dice_overlap(Y_pred, Y_batch)
        IoU_score = IoU(Y_pred, Y_batch)
        accuracy_score = accuracy(Y_pred, Y_batch)
        sensitivity_score = sensitivity(Y_pred, Y_batch)
        specificity_score = specificity(Y_pred, Y_batch)

        return loss, dice_overlap_score, IoU_score, accuracy_score, sensitivity_score, specificity_score
        
    def train(self):
        # extract parameters from config
        num_epochs = self.config["train_kwargs"]["num_epochs"]

        for epoch in range(num_epochs):
            print('* Epoch %d/%d' % (epoch+1, num_epochs))

            avg_loss = 0
            avg_dice_overlap_score = 0
            avg_IoU_score = 0
            avg_accuracy_score = 0
            avg_sensitivity_score = 0
            avg_specificity_score = 0

            self.model.train()  # train mode
            for X_batch, Y_batch in self.train_loader:
                loss, dice_overlap_score, IoU_score, accuracy_score, sensitivity_score, specificity_score = self.train_step(X_batch, Y_batch)

                # calculate metrics to show the user
                avg_loss += loss / len(self.train_loader)
                avg_dice_overlap_score += dice_overlap_score / len(self.train_loader)
                avg_IoU_score += IoU_score / len(self.train_loader)
                avg_accuracy_score += accuracy_score / len(self.train_loader)
                avg_sensitivity_score += sensitivity_score / len(self.train_loader)
                avg_specificity_score += specificity_score / len(self.train_loader)

            
            
            if self.wandb_run is not None:
                pil_image = self.test_images(validation = True)
                example = wandb.Image(pil_image, caption=f"Epoch {epoch}")
                self.wandb_run.log({
                    "train_loss": avg_loss,
                    "epoch": epoch,
                    "example": example,
                    "dice_overlap_score": avg_dice_overlap_score,
                    "IoU_score": avg_IoU_score,
                    "accuracy_score": avg_accuracy_score,
                    "sensitivity_score": avg_sensitivity_score,
                    "specificity_score": avg_specificity_score
                })
            else:
                print(' - loss: %f' % avg_loss)

    def test_images(self, validation = True):
        if validation:
            X_test, Y_test = next(iter(self.val_loader))
        else:
            X_test, Y_test = next(iter(self.test_loader))
        
        # show intermediate results
        self.model.eval()  # testing mode
        Y_hat = F.sigmoid(self.model(X_test.to(self.device))).detach().cpu()
        
        fig, axes = plt.subplots(2, 6, figsize=(20, 5))
        for k in range(6):
            ax = axes[0, k]
            ax.imshow(np.rollaxis(X_test[k].numpy(), 0, 3), cmap='gray')
            ax.set_title('Real')
            ax.set_axis_off()

            ax = axes[1, k]
            ax.imshow(Y_hat[k, 0], cmap='gray')
            ax.set_title('Output')
            ax.set_axis_off()
        # plt.suptitle('%d / %d - loss: %f' % (epoch+1, num_epochs, avg_loss))
        # plt.savefig("test.png")
        # img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        # img = from_canvas(fig)
        import io
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')

        img = PIL.Image.open(img_buf)
            
        return img
            
    def predict(self, data):
        self.model.eval()  # testing mode
        Y_pred = [F.sigmoid(self.model(X_batch.to(self.device))) for X_batch, _ in data]
        return np.array(Y_pred)
        
def from_canvas(fig):
    lst = list(fig.canvas.get_width_height())
    lst.append(3)
    return PIL.Image.fromarray(np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8).reshape(lst))

if __name__ == "__main__":
    segmentator = Segmentator(use_wandb=True)
    segmentator.train()
    
    
    # images, labels = next(iter(segmentator.train_loader))

    # for i in range(6):
    #     plt.subplot(2, 6, i+1)
    #     plt.imshow(np.swapaxes(np.swapaxes(images[i], 0, 2), 0, 1))

    #     plt.subplot(2, 6, i+7)
    #     plt.imshow(labels[i].squeeze())
    # plt.savefig("test.png")