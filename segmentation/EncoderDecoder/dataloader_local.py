import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import glob
import PIL.Image as Image

# pip install torchsummary
import torch
from torch.utils.data import  random_split
import torchvision.transforms as transforms

data_path_drive = '/mnt/c/Users/frede/Downloads/DRIVE/training'
data_path_lesion = '/mnt/c/Users/frede/Downloads/PH2_Dataset_images/'

class Lesion_Data(torch.utils.data.Dataset):
    def __init__(self, train_transform_size=128, test_transform_size=128, data_path=data_path_lesion):
        'Initialization'
        self.image_paths = sorted(glob.glob(data_path + '/***/**_Dermoscopic_Image/*.bmp'))
        self.mask_paths = sorted(glob.glob(data_path + '/***/**_lesion/*.bmp'))
        self.train_transform = transforms.Compose([transforms.Resize((train_transform_size, train_transform_size)),
                                                            transforms.ToTensor()])
        self.test_transform = transforms.Compose([transforms.Resize((test_transform_size, test_transform_size)),
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
        Y = self.train_transform(mask)
        X = self.train_transform(image)
        return X, Y

    def get_datasets(self):
        train_size = int(0.8 * len(self))
        test_size = len(self) - train_size
        val_size = int(0.2 * train_size)
        train_size = train_size - val_size
        train_dataset, test_dataset, val_dataset = random_split(self, [train_size, test_size, val_size])
        train_dataset.dataset.transform = self.train_transform
        test_dataset.dataset.transform = self.test_transform
        val_dataset.dataset.transform = self.test_transform
        return train_dataset, test_dataset, val_dataset


class DRIVE_data(torch.utils.data.Dataset):
    def __init__(self, train_transform_size=128, test_transform_size=128, data_path=data_path_drive):
        'Initialization'
        self.image_paths = sorted(glob.glob(data_path + '/images/*.tif'))
        self.mask_paths = sorted(glob.glob(data_path + '/1st_manual/*.gif'))



        self.train_transform = transforms.Compose([transforms.Resize((train_transform_size, train_transform_size)),
                                                            transforms.ToTensor()])
        self.test_transform = transforms.Compose([transforms.Resize((test_transform_size, test_transform_size)),
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
        Y = self.train_transform(mask)
        X = self.train_transform(image)
        return X, Y

    def get_datasets(self):
        train_size = int(0.8 * len(self))
        test_size = len(self) - train_size
        val_size = int(0.2 * train_size)
        train_size = train_size - val_size
        train_dataset, test_dataset, val_dataset = random_split(self, [train_size, test_size, val_size])
        train_dataset.dataset.transform = self.train_transform
        test_dataset.dataset.transform = self.test_transform
        val_dataset.dataset.transform = self.test_transform
        return train_dataset, test_dataset, val_dataset
