import torch.nn as nn

def bce_loss(y_real, y_pred, clip = False):
    m = nn.Sigmoid()
    loss = nn.BCELoss()
    return loss(m(y_pred), y_real)

loss_functions = {
    "bce": bce_loss
}