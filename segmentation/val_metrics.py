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
    _, y_true_argmax = torch.max(y_true.float(), dim=1)
    _, y_pred_argmax = torch.max(y_pred.float(), dim=1)
    correct_predictions = torch.eq(y_true_argmax, y_pred_argmax).float()
    accuracy_score = torch.mean(correct_predictions).item()
    return accuracy_score
