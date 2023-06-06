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

# ########################## Another one ########################################

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Sequential(
#                         nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
#                         nn.BatchNorm2d(out_channels),
#                         nn.ReLU())
#         self.conv2 = nn.Sequential(
#                         nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
#                         nn.BatchNorm2d(out_channels))
#         self.downsample = downsample
#         self.relu = nn.ReLU()
#         self.out_channels = out_channels

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.conv2(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out
# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes = 10):
#         super(ResNet, self).__init__()
#         self.inplanes = 64
#         self.conv1 = nn.Sequential(
#                         nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
#                         nn.BatchNorm2d(64),
#                         nn.ReLU())
#         self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
#         self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
#         self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
#         self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
#         self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.fc = nn.Linear(512, num_classes)

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes:

#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(planes),
#             )
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)


#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.maxpool(x)
#         x = self.layer0(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

#         return x


#         ##############################



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
            nn.Dropout(0.5),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
        )

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
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x


    @property
    def name(self):
        return "TestCNN"
    
    
class LukasCNN(nn.Module):
    def __init__(self, dropout = 0.5, batchnorm = True):
        super(LukasCNN, self).__init__()
        
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
        return "LukasCNN"





models = {
    "SimpleCNN": SimpleCNN,
    "ResNet": ResNet,
    "TestCNN": TestCNN,
    "LukasCNN": LukasCNN,
}




if __name__ == "__main__":
    # model = SimpleCNN()
    # model = ResNet()
    model = TestCNN()
    print(model.name)
