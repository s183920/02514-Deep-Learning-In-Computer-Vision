import pickle
import torch
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'gan/code/stylegan2-ada-pytorch-main/'))
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import PIL

# own modules
from project import run_projection
from align import align_face

class LatentExplorer:
    def __init__(self, name) -> None:
        with open('gan/pretrained_models/ffhq.pkl', 'rb') as f:
            self.G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
            
        self.name = name
        self.folder = f"gan/results/{name}"
        os.makedirs(self.folder, exist_ok=True)
            
            
    def reconstruct(self, img_path, align=True):
        if align:
            img_path = align_face(img_path)
            
        run_projection(img_path, outdir=self.folder+ "/reconstruction")
    
    
if __name__ == "__main__":
    le = LatentExplorer("Barbie")
    
    img = "gan/test_imgs/RyanGosling_Barbie.png"
    le.reconstruct(img)