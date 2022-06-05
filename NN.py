from tokenize import Double
import torch.nn as nn
import torch
import torch.nn.functional as F
from zmq import device
import numpy as np
from torch.nn.functional import conv2d
from createdata import step, pts
from torchvision.models import resnet18

PTS = torch.from_numpy(pts).float()

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self,x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)    

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up2(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
    
    def forward(self, x1):
        x1 = self.up(x1)
        
        return self.conv(x1)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class LinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.linear_layer(x)


class NeuralNet(nn.Module):
    def __init__(self, bilinear=False, device="cuda"):
        super(NeuralNet, self).__init__()
        self.bilinear = bilinear
        self.device = device

        # Unet
        self.inc = DoubleConv(4,64)
        self.d1 = Down(64, 128)
        self.d2 = Down(128, 256)
        self.d3 = Down(256, 512)
        self.d4 = Down(512, 1024) 
        self.d5 = Down(1024,1024)

        self.u1 = Up(1024+1024, 1024)
        self.u2 = Up(1024+512, 512)
        self.u3 = Up(512+256, 256)
        self.u4 = Up(256+128, 128)
        self.u5 = Up(128+64,64)
        self.out = OutConv(64, 1)
        
    def forward(self, sdf):    # [B,1,256,256] [B,1,256,256]
        B = sdf.shape[0]
        dim = sdf.shape[-1]
        pts = PTS.to(self.device).unsqueeze(0).expand(B,-1,-1)      # [B,dim*dim,3]
        pts = pts.view(B,dim,dim,3).permute(0,3,1,2)                # [B,3,dim,dim]
        x = torch.cat((pts,sdf), dim=1)                             # [B,4,dim,dim]
  
        d = self.inc(x)         # [B,64,256,256]
        d1 = self.d1(d)         # [B,128,128,128]
        d2 = self.d2(d1)        # [B,256,64,64]
        d3 = self.d3(d2)        # [B,512,32,32]
        d4 = self.d4(d3)        # [B,1024,16,16]
        d5 = self.d5(d4)        # [B,1024,8,8]

        u1 = self.u1(d5,d4)     # [B,1024,16,16]
        u2 = self.u2(u1,d3)     # [B,512,32,32]
        u3 = self.u3(u2,d2)     # [B,256,64,64]
        u4 = self.u4(u3,d1)     # [B,128,128,128]
        u5 = self.u5(u4,d)      # [B,64,256,256]
        df_pred = self.out(u5)  # [B,1,256,256]

        # first order schemes
        # dx_kernel = torch.zeros(3,3,dtype=df_pred.dtype, device= self.device)
        # dx_kernel[1] = torch.tensor([0, -1, 1], dtype=df_pred.dtype, device= self.device)/d_step
        # dx_kernel = dx_kernel.view(1, 1, 3, 3)

        # scond order schemes
        dx_kernel = torch.zeros(5,5,dtype=df_pred.dtype, device= self.device)
        dx_kernel[2] = torch.tensor([1/12, -2/3, 0, 2/3, -1/12], dtype=df_pred.dtype, device= self.device)/step
        dx_kernel = dx_kernel.view(1, 1, 5, 5)

        dy_kernel = dx_kernel.transpose(2,3)
        f_x = conv2d(df_pred, dx_kernel)
        f_y = conv2d(df_pred, dy_kernel)
        comb = torch.cat([f_y, f_x], 1) # [B, 2, H, W]
        f_norm = torch.norm(comb, dim=1)
    
        return df_pred, f_norm, f_x, f_y