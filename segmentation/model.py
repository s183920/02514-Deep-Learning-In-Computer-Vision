import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet2Exercise(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        pool_kernel_size = 2
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.Conv2d(64, 64, pool_kernel_size, stride=2)  # 128 -> 64
        # self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.Conv2d(64, 64, pool_kernel_size, stride=2)  # 64 -> 32
        # self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.Conv2d(64, 64, pool_kernel_size, stride=2)  # 32 -> 16
        # self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        # self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8
        self.pool3 = nn.Conv2d(64, 64, pool_kernel_size, stride=2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        # self.upsample0 = nn.Upsample(16)  # 8 -> 16
        self.upsample0 = nn.ConvTranspose2d(64, 64, 2, stride = 2)
        self.dec_conv0 = nn.Conv2d(64*2, 64, 3, padding=1)
        self.upsample1 = nn.ConvTranspose2d(64, 64, 2, stride = 2)
        # self.upsample1 = nn.Upsample(32)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(64*2, 64, 3, padding=1)
        self.upsample2 = nn.ConvTranspose2d(64, 64, 2, stride = 2)
        # self.upsample2 = nn.Upsample(64)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(64*2, 64, 3, padding=1)
        self.upsample3 = nn.ConvTranspose2d(64, 64, 2, stride = 2)
        # self.upsample3 = nn.Upsample(128)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(64*2, 1, 3, padding=1)

    def forward(self, x):
        # obs - we split the pooling and the conv layers to be able to concatenate them later in the skip connection
        # encoder
        e0 = F.relu(self.enc_conv0(x))
        e01 = self.pool0(e0)
        e1 = F.relu(self.enc_conv1(e01))
        e11 = self.pool1(e1)
        e2 = F.relu(self.enc_conv2(e11))
        e21 = self.pool2(e2)
        e3 = F.relu(self.enc_conv3(e21))
        e31 = self.pool3(e3)
        

        # bottleneck
        b = F.relu(self.bottleneck_conv(e31))

        # decoder
        # u0 = self.upsample0(b)
        d0 = F.relu(self.dec_conv0(torch.cat([self.upsample0(b), e3], 1)))
        d1 = F.relu(self.dec_conv1(torch.cat([self.upsample1(d0), e2], 1)))
        d2 = F.relu(self.dec_conv2(torch.cat([self.upsample2(d1), e1], 1)))
        d3 = self.dec_conv3(torch.cat([self.upsample3(d2), e0], 1))  # no activation
        return d3
    
    @property
    def name(self):
        return "UNet2Exercise"
    

models = {
    "UNet2Exercise": UNet2Exercise
}