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
import glob
import PIL.Image as Image

class DRIVE2(torch.utils.data.Dataset):
    def __init__(self, train = True, transform = None, data_augmentation = True, *args, **kwargs):
        # set datadir
        # self.datadir = 'hotdog/small_data/' + ('train' if train else 'test')
        self.data_path = '/dtu/datasets1/02514/DRIVE/' + ('training' if train else 'test')
        # self.data_path = 'KATRINE_DATA/small_data/' + ('training' if train else 'test')

        # self.data_path = '/dtu/datasets1/02514/DRIVE/training'
        self.transform = transform if transform else self.default_transform
        self.img_size = (128, 128)

        self.train = train
        self.data_augmentation = data_augmentation
        self.image_paths = sorted(glob.glob(self.data_path + '/images/*.tif'))
        self.mask_paths = sorted(glob.glob(self.data_path + '/mask/*.gif'))

        # call super
        # super().__init__(self.datadir, transform=transform, *args, **kwargs)

        # split train and val

    def default_transform(self):
        return transforms.Compose([transforms.Resize((128,128)),
                transforms.ToTensor()])

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path)
        mask = Image.open(mask_path)
        Y = self.transform(mask)
        X = self.transform(image)
        return X, Y

    def get_subsets(self):
        if self.train:
            self.train_subset, self.val_subset, self.test_subset = torch.utils.data.random_split(
        self, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(1))





    @property
    # def default_transform(self):
    #     if self.train and self.data_augmentation:

    #         return transforms.Compose([
    #             transforms.Resize(self.img_size),
    #             transforms.ToTensor()])
    #     else:
    #         return transforms.Compose([
    #             transforms.Resize(self.img_size),
    #             transforms.ToTensor()])



    def get_dataloader(self, batch_size = 32, *args, **kwargs):
        # if self.train:
        self.get_subsets()
        train_loader = DataLoader(dataset=self.train_subset, shuffle=True, batch_size=batch_size, *args, **kwargs)
        val_loader = DataLoader(dataset=self.val_subset, shuffle=False, batch_size=batch_size, *args, **kwargs)
        test_loader = DataLoader(dataset=self.test_subset, shuffle=False, batch_size=batch_size, *args, **kwargs)

        return train_loader, val_loader, test_loader
        # else:
        #     return DataLoader(self, batch_size=batch_size, shuffle=False, *args, **kwargs)
        # DataLoader(testset, batch_size=batch_size, shuffle=False)


    def transform_label(self, label):
        return self.classes[label]


if __name__ == "__main__":
    dataset = DRIVE2()
    train_loader, val_loader, test_loader = dataset.get_dataloader(batch_size=32, shuffle=True)
    images, labels = next(iter(train_loader))
    plt.figure(figsize=(20,10))

    for i in range(21):
        plt.subplot(5,7,i+1)
        plt.imshow(images[i].numpy().transpose(1,2,0))
        plt.title(dataset.transform_label(labels[i].item()))
        plt.axis('off')

    plt.savefig('DRIVE2_overview.png')
