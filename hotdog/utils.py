import numpy as np
import wandb
import torch.optim as optim
import torch

def inverse_normalize(tensor, mean = [0.5225634,0.44118169,0.35845589], std = [0.22521636,0.22928182,0.233647]):
    """
    Applies the inverse normalization transform to a PyTorch tensor.
    
    Args:
        tensor (torch.Tensor): The tensor to be transformed.
        mean (list or tuple): The mean used for normalization.
        std (list or tuple): The standard deviation used for normalization.
    
    Returns:
        torch.Tensor: The transformed tensor.
    """
    return (tensor * torch.tensor(std).reshape(1, -1, 1, 1)) + torch.tensor(mean).reshape(1, -1, 1, 1)

def data_to_img_array(data):
    return (data.numpy().transpose(0,2,3,1)*255).astype(np.int16)

def download_model(full_name:str, download_path:str = None):
    """
    Downloads the agent from wandb.
    
    Parameters
    ----------
    full_name: str
        The full name of the wandb artifact to download.
    download_path: str
        The path to download the artifact to. The agent will be downloaded to `download_path/model.zip`.
    
    Returns
    -------
    model_path: str
        The path to the downloaded agent. This is `download_path/model`.
    """
    api = wandb.Api() # start wandb api
    artifact = api.artifact(full_name) # load artifact
    path = artifact.download(download_path) # download artifact
 
    return path 


class ReduceLROnPlateau(optim.lr_scheduler.ReduceLROnPlateau):
    def get_last_lr(self):
        return self._last_lr
