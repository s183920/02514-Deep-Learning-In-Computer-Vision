import numpy as np
import wandb


def data_to_img_array(data):
    return (data.numpy().transpose(0,2,3,1)*255).astype(np.uint8)

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
 
    return path + "/model"
