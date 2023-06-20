from typing import Iterator
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torchvision

class Resnet(nn.Module):
    def __init__(self, finetune = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.finetune = finetune
        
        # self.resnet = torchvision.models.resnet50(pretrained = True)
        self.resnet = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        
        if not self.finetune:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # construct fc layer
        num_ftrs = self.resnet.fc.in_features        
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,100),
            nn.ReLU(),
            nn.Linear(100,29),
        )
        
        # replace fc layer
        self.resnet.fc = self.fc
        
        # add sigmoid layer
        self.classifier = nn.Sequential(
            nn.LogSoftmax(dim=1)
        )
        
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if self.finetune:
            return super().parameters(recurse)
        else:
            return self.resnet.fc.parameters(recurse) 
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)
        return x
    
    def set_model(self, model):
        print(f"Setting model to {model}")
        
        model = models.get(model.lower())
        if model is None:
            raise ValueError(f"Model not found")
      
        self.model = model(finetune = self.config["finetune"])

        self.model.to(self.device)
    @property 
    def name(self):
        return "Resnet"
    
models = {
    "resnet" : Resnet,
}