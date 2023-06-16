from typing import Callable, Optional
import torchvision.datasets as datasets
import numpy as np
import torch
import torchvision.transforms as transforms
import os
import json
import PIL
import matplotlib.pyplot as plt


class TacoDataset(torch.utils.data.Dataset):
    """
    Class to store the food data
    """
    root_dir = '/dtu/datasets1/02514/data_wastedetection/'
    anns_file_path = root_dir + '/' + 'annotations.json'
    
    def __init__(self, datatype = "train"):
        self.datatype = datatype
        
        # Read annotations
        with open(self.anns_file_path, 'r') as f:
            self.dataset = json.loads(f.read())
        
        self.categories = self.dataset['categories']
        self.anns = self.dataset['annotations']
        self.imgs = self.dataset['images']
        
        # split into train and test
        idxs = np.arange(len(self.imgs))
        idxs = np.random.permutation(idxs)
        self.train_idxs = idxs[:int(0.8*len(idxs))]
        self.test_idxs = idxs[int(0.8*len(idxs)):]
        self.train_idxs = self.train_idxs[:int(0.8*len(self.train_idxs))]
        self.val_idxs = self.train_idxs[int(0.8*len(self.train_idxs)):]
        
        print(f"Number of train images: {len(self.train_idxs)}")
        print(f"Number of val images: {len(self.val_idxs)}")
        print(f"Number of test images: {len(self.test_idxs)}")
        
        
        self.transform = transforms.Compose([
            transforms.PILToTensor(),
        ])
        
    def __getitem__(self, idx):
        if self.datatype == "train":
            idx = self.train_idxs[idx]
        elif self.datatype == "val":
            idx = self.val_idxs[idx]
        elif self.datatype == "test":
            idx = self.test_idxs[idx]
        
        # get data
        img_meta = self.imgs[idx]
        img_ann = self.anns[idx]
        # img = np.load(self.root_dir + img_meta['file_name'])
        img = PIL.Image.open(self.root_dir + img_meta['file_name'])
        img = self.transform(img)
        
        return img, img_meta, img_ann
        # return img_meta

# class TacoDataset(datasets.CocoDetection):    
#     def __init__(self, root: str, annFile: str, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, transforms: Callable[..., Any] | None = None) -> None:
#         self.root = '/dtu/datasets1/02514/data_wastedetection/'
#         self.annFile = self.root + '/' + 'annotations.json'
#         super().__init__(root, annFile, transform, target_transform, transforms)

def show_img(img, ax = None):
    if ax is None:
        fig, ax = plt.subplots()
        
    ax.imshow(img.permute(1, 2, 0))
    ax.axis('off')
    
    return ax

if __name__ == "__main__":

    dataset = TacoDataset()
    img, img_meta, img_ann = dataset.__getitem__(0)

    show_img(img)
    plt.savefig("test.png")