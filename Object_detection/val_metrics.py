import torch
import torchvision


def recall_score(y_true, y_pred):
    TP = torch.sum(y_true * y_pred)
    FN = torch.sum(y_true * (1 - y_pred))
    return TP / (TP + FN)

def precision_score(y_true, y_pred):
    TP = torch.sum(y_true * y_pred)
    FP = torch.sum((1 - y_true) * y_pred)
    return TP / (TP + FP)

def IoU(y_true, y_pred):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    return intersection / union

def AP_score(y_true, y_pred):
    '''
    Mean Average Precision
    
    input: 
        y_true: ground truth
        y_pred: prediction

    output:
        mAP: Mean Average Precision
    '''
    # sort y_pred by confidence
    y_pred = y_pred[torch.argsort(y_pred[:, 0], descending=True)]
    # get TP and FP
    TP = torch.cumsum(y_pred[:, 1], dim=0)
    FP = torch.cumsum(1 - y_pred[:, 1], dim=0)
    # get recall and precision
    recall = TP / (TP + FP)
    precision = TP / torch.sum(y_true)
    # get AP
    AP = torch.sum((recall[1:] - recall[:-1]) * precision[1:])
    return AP
