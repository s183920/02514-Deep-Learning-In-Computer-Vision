import torch.optim as optim

class ReduceLROnPlateau(optim.lr_scheduler.ReduceLROnPlateau):
    def get_last_lr(self):
        return self._last_lr
    
def get_optimiser(optimiser, model, **kwargs):
    if optimiser.lower() == "sgd" and "lr" not in kwargs.keys():
        lr = 0.01
        
    return optim.__dict__.get(optimiser)(model.parameters(), **kwargs)