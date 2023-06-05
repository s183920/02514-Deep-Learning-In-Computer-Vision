import torch.nn as nn




class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        
        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=224, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=224, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(16*224*224, 2),
            nn.LogSigmoid()
        )
    
    def forward(self, x):
        x = self.convolutional(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
    @property
    def name(self):
        return "SimpleCNN"
    

class NewCNN(nn.Module):
    def __init__(self):
        super(NewCNN, self).__init__()
        
        
        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=224, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=224, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(16*224*224, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.LogSigmoid()
        )
    
    def forward(self, x):
        x = self.convolutional(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
    @property
    def name(self):
        return "NewCNN"
    
    
models = {
    "SimpleCNN": SimpleCNN,
    "NewCNN": NewCNN,
}


if __name__ == "__main__":
    model = SimpleCNN()
    print(model.name)