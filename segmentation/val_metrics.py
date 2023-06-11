import torch
import torch.nn as nn

def dice_overlap(y_true, y_pred):
 #   intersection = torch.sum(y_true * y_pred, dim=[1, 2, 3])
  #  dice = (2.0 * intersection + 1e-6) / (torch.sum(y_true, dim=[1, 2, 3]) + torch.sum(y_pred, dim=[1, 2, 3]) + 1e-6)
   # dice_score = torch.mean(dice)
    #return dice_score
    Iou = IoU(y_true, y_pred)
    dice_score = 2*Iou/(1+Iou)
    return dice_score

def IoU(y_true, y_pred):
    intersection = torch.sum(y_true * y_pred, dim=[1, 2, 3])
    union = torch.sum(y_true + y_pred, dim=[1, 2, 3]) - intersection
    iou_score = torch.mean((intersection + 1e-6) / (union + 1e-6))
    return iou_score

def sensitivity(y_true, y_pred):
    true_positives = torch.sum(y_true * y_pred, dim=[1, 2, 3])
    actual_positives = torch.sum(y_true, dim=[1, 2, 3])
    sensitivity_score = torch.mean(true_positives / (actual_positives + 1e-6))
    return sensitivity_score

def specificity(y_true, y_pred):
    true_negatives = torch.sum((1 - y_true) * (1 - y_pred), dim=[1, 2, 3])
    actual_negatives = torch.sum(1 - y_true, dim=[1, 2, 3])
    specificity_score = torch.mean(true_negatives / (actual_negatives + 1e-6))
    return specificity_score

def accuracy(y_true, y_pred):
    _, predicted = torch.max(y_pred.data, 1)
    total_train = y_true.nelement()
    correct_train = predicted.eq(y_true.data).sum().item()
    accuracy_score = 100 * correct_train / total_train
    return accuracy_score
