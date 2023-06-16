import torch.nn as nn
import torch

def bce_loss(y_real, y_pred):
    # m = nn.Sigmoid()
    loss = nn.BCELoss()
    return loss(y_pred, y_real)

def cross_entropy_loss(y_real, y_pred):
    # weights = torch.tensor([0.1, 0.9])
    loss = nn.CrossEntropyLoss()
    return loss(y_pred, y_real)

def focal_loss(y_real, y_pred, gamma = 2):
    X = y_real
    Y = y_pred
    
    # focal = torch.sum((1-Y)**gamma * X * torch.log(Y) + (1-X) * torch.log(1-Y), dim = (-2, -1))
    loss = - torch.mean((1-Y)**gamma * X * torch.log(Y + 1e-8) + (1-X) * torch.log(1-Y + 1e-8))
    
    # return -torch.mean(focal)
    return loss
    
def total_variation(y_real, y_pred):
    # Compute the total variation of the predicted image
    diff_i = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    diff_j = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
    tv = torch.mean(diff_i) + torch.mean(diff_j)
    return tv

def bce_total_variation(y_real, y_pred):
    bce = bce_loss(y_real, y_pred)
    tv = total_variation(y_real, y_pred)
    return bce + 0.1*tv


def dice_loss(y_real, y_pred):
    nom = torch.mean(2*y_real*y_pred + 1, dim = (-2, -1))
    denom = torch.mean(y_real + y_pred, dim = (-2, -1)) + 1
    return torch.mean(1-nom / denom)

def sensitivity(y_real, y_pred):
    # true positives / (true positives + false negatives)
    # tp = torch.mean(y_real * y_pred, dim = (-2, -1))
    # fn = torch.sum(y_real * (1-y_pred), dim = (-2, -1))
    return - torch.mean(y_real*y_pred)

def dice_sensitive_loss(y_real, y_pred):
    dice = dice_loss(y_real, y_pred)
    sens = sensitivity(y_real, y_pred)
    return dice + sens

loss_functions = {
    "bce": bce_loss,
    "cross_entropy": cross_entropy_loss,
    "focal": focal_loss,
    "bce_tv": bce_total_variation,
    "dice": dice_loss,
    "dice_sensitivity": dice_sensitive_loss,
}