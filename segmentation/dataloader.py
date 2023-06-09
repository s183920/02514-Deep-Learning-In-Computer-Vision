import torch
import os
import glob
import PIL.Image as Image
from torchvision import transforms

class PhC(torch.utils.data.Dataset):
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
    
datasets = {
    "PhC": PhC
}