import torch
import numpy as np

def inverse_normalize(tensor, dataset):
    """
    Applies the inverse normalization transform to a PyTorch tensor.
    
    Args:
        tensor (torch.Tensor): The tensor to be transformed.
        mean (list or tuple): The mean used for normalization.
        std (list or tuple): The standard deviation used for normalization.
    
    Returns:
        torch.Tensor: The transformed tensor.
    """
    
    if dataset == "Lesion":
        mean = torch.tensor([0.75376641, 0.57684034, 0.48878668])
        std = torch.tensor([0.15750518, 0.15277521, 0.15188558])
    elif dataset == "DRIVE":
        mean = torch.tensor([0.49740665, 0.27065086, 0.16243291])
        std = torch.tensor([0.32961248, 0.17564391, 0.09661924])
        
    return (tensor * std.reshape(1, -1, 1, 1)) + mean.reshape(1, -1, 1, 1)

def data_to_img_array(data):
    return (data.numpy().transpose(0,2,3,1)*255).astype(np.int16)
