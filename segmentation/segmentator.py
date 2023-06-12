import torch
import torch.nn.functional as F
from tools import Agent
from dataloader import get_datasets
from model import models
from loss_functions import loss_functions
from val_metrics import Scorer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from hparams import default_config
import PIL
import wandb
import io


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
        scores = Scorer(Y_batch, Y_pred, return_method="sum").get_scores()

        
        return loss, scores
        
    def train(self):
        # extract parameters from config
        num_epochs = self.config["num_epochs"]
        validation_metric = self.config["validation_metric"]
        best_score = 0

        for epoch in range(num_epochs):
            print('* Epoch %d/%d' % (epoch+1, num_epochs))

            avg_loss = 0
            avg_train_scores = {"dice_overlap": 0, "IoU": 0, "accuracy": 0, "sensitivity": 0, "specificity": 0}
            self.model.train()  # train mode
            for X_batch, Y_batch in self.train_loader:
                loss, scores = self.train_step(X_batch, Y_batch)

                # calculate metrics to show the user
                avg_loss += loss / len(self.train_loader)
                avg_train_scores = {k: avg_train_scores[k] + v / len(self.trainset) for k, v in scores.items()}
            
            # validation
            val_loss, val_scores = self.test(validation = True)
            print(f"Validation loss: {val_loss}")
                
            # Save model
            if self.config["model_save_freq"] is None:
                if val_loss < best_score:
                    best_score = val_loss
                    self.save_model(f"logs/{self.project}/models/{self.name}", "model.pth", f"model saved with validation loss {best_score:.3f} on {self.config['dataset']} data at epoch {epoch}")
            elif epoch % self.config["model_save_freq"] == 0:
                self.save_model(f"logs/{self.project}/models/{self.name}", "model.pth", f"model saved with validation loss {best_score:.3f} on {self.config['dataset']} data at epoch {epoch}")
            
            if self.wandb_run is not None:
                imgs = self.test_images(validation = True, for_wandb = True)
                # examples = [wandb.Image(img, caption=f"Idx {idx}") for idx, img in enumerate(imgs)]
                self.wandb_run.log({
                    "Loss/train_loss": avg_loss,
                    "Loss/val_loss": val_loss,
                    "Epoch": epoch,
                    f"{self.config['dataset']} validation examples": wandb.Image(imgs),
                    **{"Train scores/" + k: v for k, v in avg_train_scores.items()},
                    **{"Validation scores/" + k: v for k, v in val_scores.items()},
                    "WANDB segmentation": self.wandb_test_segmentation(validation=True),
                })
            else:
                print(' - loss: %f' % avg_loss)
        
        # clear cache
        self.clear_cache()
        
    def test(self, validation = False):
        if validation:
            data_loader = self.val_loader
        else:
            data_loader = self.test_loader
        
        loader_len = len(data_loader)    
        data_len = len(data_loader.dataset)
        
        # init counters
        test_scores = {"dice_overlap": 0, "IoU": 0, "accuracy": 0, "sensitivity": 0, "specificity": 0}
        test_loss = 0
        
        # test model
        self.model.eval()
        for X_pred, Y_true in data_loader:
            X_pred = X_pred.to(self.device)
            with torch.no_grad():
                Y_pred = self.model(X_pred).cpu()
                
            # update counters
            test_loss += self.loss_fn(Y_pred.cpu(), Y_true).item()
            test_scores = {k: test_scores[k] + v for k, v in Scorer(Y_true, Y_pred, return_method="sum").get_scores().items()}
        
        # calculate average loss and scores
        test_loss /= loader_len
        test_scores = {k: v / data_len for k, v in test_scores.items()}
        
        return test_loss, test_scores

    def test_images(self, validation = True, for_wandb = False):
        
        if validation:
            X_test, Y_test = next(iter(self.val_loader))
        else:
            X_test, Y_test = next(iter(self.test_loader))
            
        ncols = min(X_test.shape[0], 6)
        nrows = 3
        
        # show intermediate results
        self.model.eval()  # testing mode
        Y_hat = F.sigmoid(self.model(X_test.to(self.device))).detach().cpu()
        
        
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(2+3*ncols, 3*nrows))
        for k in range(ncols):
            
            # show input image
            ax = axes[0, k]
            ax.imshow(np.rollaxis(X_test[k].numpy(), 0, 3), cmap='gray')
            ax.set_title('Real')
            ax.set_axis_off()

            # show mask prediction
            ax = axes[1, k]
            ax.imshow(Y_hat[k, 0], cmap='gray')
            ax.set_title('Output')
            ax.set_axis_off()
            
            # show ground truth mask
            ax = axes[2, k]
            ax.imshow(Y_test[k, 0], cmap='gray')
            ax.set_title('Ground truth')
            ax.set_axis_off()

        if for_wandb:
            # save image
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img = PIL.Image.open(img_buf)
                
            return img
        
    def wandb_test_segmentation(self, validation = True):
        labels = {i : l for i, l in enumerate(self.trainset.dataset.classes)}
        
        if validation:
            X_test, Y_test = next(iter(self.val_loader))
        else:
            X_test, Y_test = next(iter(self.test_loader))
        self.model.eval()  # testing mode
        Y_hat = F.sigmoid(self.model(X_test.to(self.device))).detach().cpu()

        # util function for generating interactive image mask from components
        def wb_mask(bg_img, pred_mask, true_mask):
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
            return wandb.Image(bg_img, masks={
                "prediction" : {"mask_data" : pred_mask, "class_labels" : labels},
                "ground truth" : {"mask_data" : true_mask, "class_labels" : labels}})
            
        examples = [wb_mask(np.rollaxis(X_test[k].numpy(), 0, 3), Y_hat[k, 0].numpy(), Y_test[k, 0].numpy()) for k in range(3)]
        
        return examples
            
    def predict(self, data):
        self.model.eval()  # testing mode
        Y_pred = [F.sigmoid(self.model(X_batch.to(self.device))) for X_batch, _ in data]
        return np.array(Y_pred)


if __name__ == "__main__":
    segmentator = Segmentator(use_wandb=True, dataset = "Lesion", num_epochs = 3)
    segmentator.train()
    
    
    # images, labels = next(iter(segmentator.train_loader))

    # for i in range(6):
    #     plt.subplot(2, 6, i+1)
    #     plt.imshow(np.swapaxes(np.swapaxes(images[i], 0, 2), 0, 1))

    #     plt.subplot(2, 6, i+7)
    #     plt.imshow(labels[i].squeeze())
    # plt.savefig("test.png")