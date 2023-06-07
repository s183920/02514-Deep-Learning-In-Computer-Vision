from typing import Iterator
import torch.nn as nn
from torch.nn.parameter import Parameter
import torchvision


class Resnet18(nn.Module):
    def __init__(self, finetune = False, dropout = 0, batchnorm = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.finetune = finetune
        
        self.resnet18 = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        
        if not self.finetune:
            for param in self.resnet18.parameters():
                param.requires_grad = False

        # construct fc layer
        num_ftrs = self.resnet18.fc.in_features        
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,100),
            nn.ReLU(),
            nn.Linear(100,2),
        )
        
        # replace fc layer
        self.resnet18.fc = self.fc
        
        # add sigmoid layer
        self.classifier = nn.Sequential(
            nn.LogSigmoid()
        )
        
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if self.finetune:
            return super().parameters(recurse)
        else:
            return self.resnet18.fc.parameters(recurse) 
        
    def forward(self, x):
        x = self.resnet18(x)
        x = self.classifier(x)
        return x
    
    @property 
    def name(self):
        return "Resnet18"


class ResNetBlock(nn.Module):
    def __init__(self, n_features):
        super(ResNetBlock, self).__init__()
        self.ResNet_block = nn.Sequential(
            nn.Conv2d(n_features, n_features, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_features, n_features, kernel_size = 3, padding=1)
        )
    def forward(self, x):
        out = nn.functional.relu(self.ResNet_block(x) + x)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels=3, n_features =16, num_ResNet_blocks=3, block_depth = 1):
        super(ResNet, self).__init__()
        conv_layers = [
            nn.Conv2d(in_channels, n_features, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ]


        for i in range(block_depth):
            for _ in range(num_ResNet_blocks):
                conv_layers.append(ResNetBlock(n_features*(i+1)))
            conv_layers.append(nn.Conv2d(n_features*(i+1), n_features*(i+2), kernel_size=3, stride=1, padding=1))
            conv_layers.append(nn.ReLU())

        conv_layers.append(nn.AvgPool2d(2))
        self.ResNet_blocks = nn.Sequential(*conv_layers)


        self.fc = nn.Sequential(
            nn.Linear(16*16*n_features*(block_depth+1), 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512,2),
            nn.LogSigmoid()
        )

    def forward(self, x):
        x = self.ResNet_blocks(x)
        x = x.view(x.size(0), -1)    #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        out = self.fc(x)
        return out


    @property
    def name(self):
        return "ResNet"

    
class SimpleCNN(nn.Module):
    def __init__(self, dropout = 0.5, batchnorm = True):
        super(SimpleCNN, self).__init__()
        
        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.batchnorm_layer = nn.Sequential(nn.BatchNorm2d(128),) if batchnorm else nn.Sequential()

   
        self.fully_connected = nn.Sequential(
            nn.Linear(16*16*128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,100),
            nn.ReLU(),
            nn.Linear(100,2),
            nn.LogSigmoid()
        )


    def forward(self, x):
        x = self.convolutional(x)
        x = self.batchnorm_layer(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x


    @property
    def name(self):
        return "SimpleCNN"
    

class PeetzCNN(nn.Module):
    def __init__(self, dropout = 0.5, batchnorm = True):
        super(PeetzCNN, self).__init__()
        
        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=5, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 5, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.batchnorm_layer = nn.Sequential(nn.BatchNorm2d(128),) if batchnorm else nn.Sequential()

   
        self.fully_connected = nn.Sequential(
            nn.Linear(12*12*128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,100),
            nn.ReLU(),
            nn.Linear(100,2),
            nn.LogSigmoid()
        )


    def forward(self, x):
        x = self.convolutional(x)
        x = self.batchnorm_layer(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x


    @property
    def name(self):
        return "PeetzCNN"
    


class PeetzCNN(nn.Module):
    def __init__(self, dropout = 0.5, batchnorm = True):
        super(PeetzCNN, self).__init__()
        
        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=5, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 5, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.batchnorm_layer = nn.Sequential(nn.BatchNorm2d(128),) if batchnorm else nn.Sequential()

   
        self.fully_connected = nn.Sequential(
            nn.Linear(12*12*128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,100),
            nn.ReLU(),
            nn.Linear(100,2),
            nn.LogSigmoid()
        )


    def forward(self, x):
        x = self.convolutional(x)
        x = self.batchnorm_layer(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x


    @property
    def name(self):
        return "PeetzCNN"



class PeetzCNN_threee(nn.Module):
    def __init__(self, dropout = 0.5, batchnorm = True):
        super(PeetzCNN_threee, self).__init__()
        
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
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.batchnorm_layer = nn.Sequential(nn.BatchNorm2d(128),) if batchnorm else nn.Sequential()

   
        self.fully_connected = nn.Sequential(
            nn.Linear(16*16*128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,100),
            nn.ReLU(),
            nn.Linear(100,2),
            nn.LogSigmoid()
        )


    def forward(self, x):
        x = self.convolutional(x)
        x = self.batchnorm_layer(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x


    @property
    def name(self):
        return "PeetzCNN_three"



class PeetzCNNLinearDropout(nn.Module):
    def __init__(self, dropout = 0.5, batchnorm = True):
        super(PeetzCNNLinearDropout, self).__init__()
        
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
            #nn.MaxPool2d(2),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.batchnorm_layer = nn.Sequential(nn.BatchNorm2d(128),) if batchnorm else nn.Sequential()

   
        self.fully_connected = nn.Sequential(
            nn.Linear(32*32*128, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256,100),
            nn.ReLU(),
            nn.Linear(100,2),
            nn.LogSigmoid()
        )


    def forward(self, x):
        x = self.convolutional(x)
        x = self.batchnorm_layer(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x


    @property
    def name(self):
        return "PeetzCNNLinearDropout"


models = {
    "SimpleCNN": SimpleCNN,
    "ResNet": ResNet,
    "Resnet18": Resnet18,
    #"TestCNN": TestCNN,
    #"LukasCNN": LukasCNN,
    "PeetzCNN": PeetzCNN,
    "PeetzCNN_three": PeetzCNN_threee,
    "PeetzCNNLinearDropout": PeetzCNNLinearDropout
}




if __name__ == "__main__":
    model = SimpleCNN()
    # model = ResNet()
    # model = TestCNN()
    print(model.name)
