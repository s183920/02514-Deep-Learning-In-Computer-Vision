import torch
import numpy as np

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
