import torch
import torch.nn as nn

# def dice_overlap(y_true, y_pred):
#  #   intersection = torch.sum(y_true * y_pred, dim=[1, 2, 3])
#   #  dice = (2.0 * intersection + 1e-6) / (torch.sum(y_true, dim=[1, 2, 3]) + torch.sum(y_pred, dim=[1, 2, 3]) + 1e-6)
#    # dice_score = torch.mean(dice)
#     #return dice_score
#     Iou = IoU(y_true, y_pred)
#     dice_score = 2*Iou/(1+Iou)
#     return dice_score

# def IoU(y_true, y_pred):
#     intersection = torch.sum(y_true * y_pred, dim=[1, 2, 3])
#     union = torch.sum(y_true + y_pred, dim=[1, 2, 3]) - intersection
#     iou_score = torch.mean((intersection + 1e-6) / (union + 1e-6))
#     return iou_score

# def sensitivity(y_true, y_pred):
#     true_positives = torch.sum(y_true * y_pred, dim=[1, 2, 3])
#     actual_positives = torch.sum(y_true, dim=[1, 2, 3])
#     sensitivity_score = torch.mean(true_positives / (actual_positives + 1e-6))
#     return sensitivity_score

# def specificity(y_true, y_pred):
#     true_negatives = torch.sum((1 - y_true) * (1 - y_pred), dim=[1, 2, 3])
#     actual_negatives = torch.sum(1 - y_true, dim=[1, 2, 3])
#     specificity_score = torch.mean(true_negatives / (actual_negatives + 1e-6))
#     return specificity_score

# def accuracy(y_true, y_pred):
#     _, y_true_argmax = torch.max(y_true.float(), dim=1)
#     _, y_pred_argmax = torch.max(y_pred.float(), dim=1)
#     correct_predictions = torch.eq(y_true_argmax, y_pred_argmax).float()
#     accuracy_score = torch.mean(correct_predictions).item()
#     return accuracy_score


class Scorer:
    def __init__(self, y_true, y_pred, class_threshold = 0.5, class_dims = 1, return_method = "mean"):
        if class_dims > 1:
            raise NotImplementedError("Not implemented for class_dims > 1, i.e. multiclass")
        
        # set attributes
        if return_method == "mean":
            self.return_method = torch.mean
        elif return_method == "sum":
            self.return_method = torch.sum
        else:
            raise NotImplementedError("Not implemented for return_method != 'mean' or 'sum'")
        
        self.y_true = y_true
        self.y_pred_vals = y_pred
        self.y_pred = (y_pred > class_threshold).int()
        
        # get confusion matrix
        self.tp = torch.sum(self.y_true * self.y_pred, dim=[1, 2, 3])
        self.fp = torch.sum((1 - self.y_true) * self.y_pred, dim=[1, 2, 3])
        self.fn = torch.sum(self.y_true * (1 - self.y_pred), dim=[1, 2, 3])
        self.tn = torch.sum((1 - self.y_true) * (1 - self.y_pred), dim=[1, 2, 3])
    
    def sensitivity(self):
        return self.return_method(self.tp / (self.tp + self.fn + 1e-6))
    
    def specificity(self):
        return self.return_method(self.tn / (self.tn + self.fp + 1e-6))
    
    def accuracy(self):
        return self.return_method((self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + 1e-6))
    
    def IoU(self):
        intersection = torch.sum(self.y_true * self.y_pred, dim=[1, 2, 3])
        union = torch.sum(self.y_true + self.y_pred, dim=[1, 2, 3]) - intersection
        return self.return_method((intersection + 1e-6) / (union + 1e-6))
    
    def dice(self):
        iou = self.IoU()
        return 2*iou/(1+iou)
    
    def get_scores(self):
        return {
            "dice_overlap": self.dice(),
            "IoU": self.IoU(),
            "accuracy": self.accuracy(),
            "sensitivity": self.sensitivity(),
            "specificity": self.specificity(),
        }
