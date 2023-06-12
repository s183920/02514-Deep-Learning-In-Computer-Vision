import torch
import os
import glob
import PIL.Image as Image
from torchvision import transforms
from torch.utils.data import  random_split
import random

class PhC(torch.utils.data.Dataset):
    classes = ['Background', 'Cell']
    
    def __init__(self, train):
        'Initialization'
        self.data_path = '/dtu/datasets1/02514/phc_data'
        # self.transform = transform
        data_path = os.path.join(self.data_path, 'train' if train else 'test')
        self.image_paths = sorted(glob.glob(data_path + '/images/*.jpg'))
        self.label_paths = sorted(glob.glob(data_path + '/labels/*.png'))
        self.train = train
        
        size = 128
        self.train_transform = transforms.Compose([transforms.Resize((size, size)), 
                                            transforms.ToTensor()])
        self.test_transform = transforms.Compose([transforms.Resize((size, size)), 
                                            transforms.ToTensor()])
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(image_path)
        label = Image.open(label_path)
        
        if self.train:
            Y = self.train_transform(label)
            X = self.train_transform(image)
        else:
            Y = self.test_transform(label)
            X = self.test_transform(image)
        return X, Y
    
class Lesion_Data(torch.utils.data.Dataset):
    data_path = '/dtu/datasets1/02514/PH2_Dataset_images'
    classes = ['Background', 'Lesion']
    
    def __init__(self, train_transform_size=128, test_transform_size=128, data_augmentation = False):
        'Initialization'
        self.image_paths = sorted(glob.glob(self.data_path + '/***/**_Dermoscopic_Image/*.bmp'))
        self.mask_paths = sorted(glob.glob(self.data_path + '/***/**_lesion/*.bmp'))
        
        if data_augmentation:
            self.train_transform = transforms.Compose([
                transforms.Resize((train_transform_size, train_transform_size)), 
                transforms.RandomRotation(random.randint(0,35)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor()
            ])
            self.test_transform = transforms.Compose([
                transforms.Resize((test_transform_size, test_transform_size)), 
                transforms.RandomRotation(random.randint(0,35)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor()
            ])
        else:
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
        Y = self.transform(mask)
        X = self.transform(image)
        return X, Y
    
    
    def get_datasets(self):
        # Split into train, test, val
        train_size = int(0.8 * len(self))
        test_size = len(self) - train_size
        val_size = int(0.2 * train_size)
        train_size = train_size - val_size
        
        # get datasets
        train_dataset, test_dataset, val_dataset = random_split(self, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(1))
        
        # set transforms
        train_dataset.dataset.transform = self.train_transform
        test_dataset.dataset.transform = self.test_transform
        val_dataset.dataset.transform = self.test_transform
        
        # set classes   
        train_dataset.dataset.classes = self.classes
        test_dataset.dataset.classes = self.classes
        val_dataset.dataset.classes = self.classes
        
        return train_dataset, test_dataset, val_dataset
    
class DRIVE_data(torch.utils.data.Dataset):
    data_path = "/dtu/datasets1/02514/DRIVE/training"
    classes = ['Background', 'Vessel']
    def __init__(self, train_transform_size=128, test_transform_size=128, data_augmentation = False):
        'Initialization'
        self.image_paths = sorted(glob.glob(self.data_path + '/images/*.tif'))
        self.mask_paths = sorted(glob.glob(self.data_path + '/1st_manual/*.gif'))



        if data_augmentation:
            self.train_transform = transforms.Compose([
                transforms.Resize((train_transform_size, train_transform_size)), 
                transforms.RandomRotation(random.randint(0,35)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor()
            ])
            self.test_transform = transforms.Compose([
                transforms.Resize((test_transform_size, test_transform_size)), 
                transforms.RandomRotation(random.randint(0,35)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor()
            ])
        else:
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
        Y = self.transform(mask)
        X = self.transform(image)
        return X, Y

    def get_datasets(self):
        # Split into train, test, val
        train_size = int(0.8 * len(self))
        test_size = len(self) - train_size
        val_size = int(0.2 * train_size)
        train_size = train_size - val_size
        train_dataset, test_dataset, val_dataset = random_split(self, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(1))
        
        # set transforms
        train_dataset.dataset.transform = self.train_transform
        test_dataset.dataset.transform = self.test_transform
        val_dataset.dataset.transform = self.test_transform
        
        # set classes   
        train_dataset.dataset.classes = self.classes
        test_dataset.dataset.classes = self.classes
        val_dataset.dataset.classes = self.classes
        
        return train_dataset, test_dataset, val_dataset
    
def get_datasets(dataset_name, **kwargs):
    if dataset_name == "PhC":
        train_dataset = PhC(train=True, **kwargs)
        test_dataset = PhC(train=False, **kwargs)
        val_dataset = None
    elif dataset_name == "Lesion":
        train_dataset, test_dataset, val_dataset = Lesion_Data(**kwargs).get_datasets()
    elif dataset_name == "DRIVE":
        train_dataset, test_dataset, val_dataset = DRIVE_data(**kwargs).get_datasets()
    else:
        raise ValueError("Dataset {} not found".format(dataset_name))
    
    return train_dataset, test_dataset, val_dataset
