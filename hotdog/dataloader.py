import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class HotdogDataset(datasets.ImageFolder):
    
    def __init__(self, train = True, transform = None, *args, **kwargs):
        # self.datadir = 'hotdog/data/' + ('train' if train else 'test')
        self.datadir = '/dtu/datasets1/02514/hotdog_nothotdog/' + ('train' if train else 'test')
        transform = transform if transform else self.default_transform
        super().__init__(self.datadir, transform=transform, *args, **kwargs)
        
    @property
    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
    def get_dataloader(self, batch_size = 32, shuffle = True, *args, **kwargs):
        # DataLoader(testset, batch_size=batch_size, shuffle=False)
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, *args, **kwargs)
    
    def transform_label(self, label):
        return self.classes[label]
    







    
if __name__ == "__main__":
    dataset = HotdogDataset()
    images, labels = next(iter(dataset.get_dataloader(batch_size=21, shuffle=False)))
    plt.figure(figsize=(20,10))

    for i in range(21):
        plt.subplot(5,7,i+1)
        plt.imshow(images[i].numpy().transpose(1,2,0))
        plt.title(dataset.transform_label(labels[i].item()))
        plt.axis('off')
    
    plt.savefig('hotdog_overview.png')

    




