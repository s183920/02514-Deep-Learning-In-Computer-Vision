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
    

class TestCNN(nn.Module):
    def __init__(self):
        super(TestCNN, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(), 
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1), 
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fully_connected = nn.Sequential(
            nn.Linear(128*(16*16), 1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.LogSoftmax(dim=1)
        )

    
    def forward(self, x):
        x = self.convolutional(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x
    
    
    @property
    def name(self):
        return "TestCNN"

     

    
models = {
    "SimpleCNN": SimpleCNN,
    "TestCNN": TestCNN
}


if __name__ == "__main__":
    #model = SimpleCNN()
    model = TestCNN()
    print(model.name)