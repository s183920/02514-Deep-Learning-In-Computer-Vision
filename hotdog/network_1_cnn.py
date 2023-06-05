from dataloader import *
import torch.nn.functional as F
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import torch.nn as nn



print("Loading datasets...")

# Load datasets
data_train = HotdogDataset()
data_test = HotdogDataset(train=False)


if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.convolutional = nn.Sequential(
            # 3 channels as input -- Colored images
            # out channels arbritrarily set
            # kenel is (5x5)
            nn.Conv2d(in_channels=3, out_channels=50, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=3),
            nn.ReLU(),
            # 2x2 reduction and stride 2 -- We move 2 pixels every time
            nn.MaxPool2d(2,2),
            # input size must mach previous output size.
            nn.Conv2d(in_channels=100, out_channels=50, kernel_size=3),
            nn.ReLU()

        )

        self.fully_connected = nn.Sequential(
            # output size from conv layer is calculated below
            nn.Linear(in_features=50*71*71, out_features=120),
            nn.Linear(in_features=120, out_features=60),
            nn.Linear(in_features=60, out_features=2),
            nn.Softmax(dim = 1)
        )

    
    def forward(self, x):
        x = self.convolutional(x)
        #print(x.size())
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x
     

print("Initializing network")
model = Network()
model.to(device)
#Initialize the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# Expand data
train_loader = data_train.get_dataloader()
test_loader = data_test.get_dataloader()
#Get the first minibatch
data = next(iter(train_loader))[0].cuda()
#Try running the model on a minibatch
print('Shape of the output from the convolutional part', model.convolutional(data).shape)
#model(data); #if this runs the model dimensions fit



num_epochs = 50

for epoch in tqdm(range(num_epochs), unit='epoch'):
    #For each epoch
    train_correct = 0
    model.train()   
    for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        #Zero the gradients computed for each weight
        optimizer.zero_grad()
        #Forward pass your image through the network
        output = model(data)
        #Compute the loss
        loss = F.nll_loss(torch.log(output), target)
        #Backward pass through the network
        loss.backward()
        #Update the weights
        optimizer.step()
        
        #Compute how many were correctly classified
        predicted = output.argmax(1)
        train_correct += (target==predicted).sum().cpu().item()
    #Comput the test accuracy
    test_correct = 0
    model.eval()
    for data, target in test_loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
        predicted = output.argmax(1).cpu()
        test_correct += (target==predicted).sum().item()
    train_acc = train_correct/len(data_train)
    test_acc = test_correct/len(data_test)
    print("Accuracy train: {train:.1f}%\t test: {test:.1f}%".format(test=100*test_acc, train=100*train_acc))


     
