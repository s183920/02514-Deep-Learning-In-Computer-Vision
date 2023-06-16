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
    
    def __init__(self):
        
        # Read annotations
        with open(self.anns_file_path, 'r') as f:
            self.dataset = json.loads(f.read())
        
        self.categories = self.dataset['categories']
        self.anns = self.dataset['annotations']
        self.imgs = self.dataset['images']
        
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def __getitem__(self, idx):
        img_meta = self.imgs[idx]
        img_ann = self.anns[idx]
        # img = np.load(self.root_dir + img_meta['file_name'])
        img = PIL.Image.open(self.root_dir + img_meta['file_name'])
        img = self.transform(img)
        
        return img, img_meta, img_ann
        # return img_meta

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