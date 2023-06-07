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
import random

class HotdogDataset(datasets.ImageFolder):

    def __init__(self, train = True, transform = None, data_augmentation = True, *args, **kwargs):
        # set datadir
        # self.datadir = 'hotdog/small_data/' + ('train' if train else 'test')
        self.datadir = '/dtu/datasets1/02514/hotdog_nothotdog/' + ('train' if train else 'test')
        
        # set values
        self.img_size = (128, 128)
        self.train = train
        self.data_augmentation = data_augmentation
        transform = transform if transform else self.default_transform
        
        # call super
        super().__init__(self.datadir, transform=transform, *args, **kwargs)
        
        # split train and val
        if self.train:
            self.train_subset, self.val_subset = torch.utils.data.random_split(
        self, [0.8, 0.2], generator=torch.Generator().manual_seed(1))
        
    @property
    def default_transform(self):
        if self.train and self.data_augmentation:
            return transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomRotation(random.randint(0,35)),
                transforms.ColorJitter(brightness=.5, hue=.3),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.ToTensor(),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ])
        
    def get_dataloader(self, batch_size = 64, *args, **kwargs):
        if self.train:
            train_loader = DataLoader(dataset=self.train_subset, shuffle=True, batch_size=batch_size, *args, **kwargs)
            val_loader = DataLoader(dataset=self.val_subset, shuffle=False, batch_size=batch_size, *args, **kwargs)
            return train_loader, val_loader
        else:
            return DataLoader(self, batch_size=batch_size, shuffle=False, *args, **kwargs)
        # DataLoader(testset, batch_size=batch_size, shuffle=False)
        
    
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
