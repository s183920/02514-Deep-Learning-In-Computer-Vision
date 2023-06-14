import pickle
import torch
import sys
import os
# sys.path.append(os.path.join(os.getcwd(), 'gan/code/stylegan2-ada-pytorch-main/torch_utils/__init__.py'))
# sys.path.append(os.path.join(os.getcwd(), 'gan/code/stylegan2-ada-pytorch-main/dnn_libs'))
sys.path.append(os.path.join(os.getcwd(), 'code/stylegan2-ada-pytorch-main/'))

with open('pretrained_models/ffhq.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
z = torch.randn([1, G.z_dim]).cuda()    # latent codes
c = None                                # class labels (not used in this example)
img = G(z, c)                           # NCHW, float32, dynamic range [-1, +1]