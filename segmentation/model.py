import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 64, 5, padding=2)
        self.enc_bn1 = nn.BatchNorm2d(64)
        self.enc_conv2 = nn.Conv2d(64, 128, 5, padding=2)
        self.enc_bn2 = nn.BatchNorm2d(128)
        self.enc_conv3 = nn.Conv2d(128, 256, 5, padding=2)
        self.enc_bn3 = nn.BatchNorm2d(256)
        self.enc_conv4 = nn.Conv2d(256, 512, 5, padding=2)
        self.enc_bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(512, 512, 5, padding=2)
        self.bottleneck_bn = nn.BatchNorm2d(512)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec_conv1 = nn.Conv2d(512, 256, 5, padding=2)
        self.dec_bn1 = nn.BatchNorm2d(256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec_conv2 = nn.Conv2d(256, 128, 5, padding=2)
        self.dec_bn2 = nn.BatchNorm2d(128)
        self.upconv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec_conv3 = nn.Conv2d(128, 64, 5, padding=2)
        self.dec_bn3 = nn.BatchNorm2d(64)
        self.dec_conv4 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.enc_bn1(self.enc_conv1(x)))
        x2 = self.pool(x1)
        x2 = F.relu(self.enc_bn2(self.enc_conv2(x2)))
        x3 = self.pool(x2)
        x3 = F.relu(self.enc_bn3(self.enc_conv3(x3)))
        x4 = self.pool(x3)
        x4 = F.relu(self.enc_bn4(self.enc_conv4(x4)))

        # Bottleneck
        bottleneck = F.relu(self.bottleneck_bn(self.bottleneck_conv(x4)))

        # Decoder
        x = F.relu(self.upconv1(bottleneck))
        x = torch.cat((x, x3), dim=1)
        x = F.relu(self.dec_bn1(self.dec_conv1(x)))
        x = F.relu(self.upconv2(x))
        x = torch.cat((x, x2), dim=1)
        x = F.relu(self.dec_bn2(self.dec_conv2(x)))
        x = F.relu(self.upconv3(x))
        x = torch.cat((x, x1), dim=1)
        x = F.relu(self.dec_bn3(self.dec_conv3(x)))
        x = self.dec_conv4(x)

        return torch.sigmoid(x)
    
    @property
    def name(self):
        return "UNet"
   
    
    
class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.bn = nn.BatchNorm2d(64)

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(16)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(32)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(64)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(128)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = self.pool0(F.relu(self.bn(self.enc_conv0(x))))
        e1 = self.pool1(F.relu(self.bn(self.enc_conv1(e0))))
        e2 = self.pool2(F.relu(self.bn(self.enc_conv2(e1))))
        e3 = self.pool3(F.relu(self.bn(self.enc_conv3(e2))))

        # bottleneck
        b = F.relu(self.bn(self.bottleneck_conv(e3)))

        # decoder
        d0 = F.relu(self.bn(self.dec_conv0(self.upsample0(b))))
        d1 = F.relu(self.bn(self.dec_conv1(self.upsample1(d0))))
        d2 = F.relu(self.bn(self.dec_conv2(self.upsample2(d1))))
        d3 = self.dec_conv3(self.upsample3(d2))  # no activation
        return torch.sigmoid(d3)
    
    @property  
    def name(self):
        return "Baseline"

models = {
    "UNet" : UNet,
    "Baseline": Baseline
}