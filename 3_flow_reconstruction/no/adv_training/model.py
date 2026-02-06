# ---------------------------------------------------------------------------------------------
# Author: Vivek Oommen
# Date: 09/01/2025
# ---------------------------------------------------------------------------------------------


import torch.nn as nn
import torch
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
from torch.nn.utils import spectral_norm

from MedicalNet.models.resnet import resnet10


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #d2l.try_gpu()
DTYPE = torch.float32

class FeatureExtractor(nn.Module):
    def __init__(self, Par, weight_path="pretrain/resnet_10_23dataset.pth", in_channels=1, layers=[1, 2, 3]):
        super().__init__()
        self.layers = layers

        self.inp_shift = Par["out_shift"].reshape(1,-1,1,1,1)
        self.inp_scale = Par["out_scale"].reshape(1,-1,1,1,1)

        # Build model and replace first conv layer if needed
        self.model = resnet10(sample_input_D=64, sample_input_H=64, sample_input_W=64, num_seg_classes=2)

        # Remove final classification layer
        self.model.fc = nn.Identity()

        # Load weights
        checkpoint = torch.load(weight_path, map_location=device)
        state_dict = checkpoint['state_dict']

        # Remove 'module.' prefix if it exists
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Now load
        self.model.load_state_dict(state_dict, strict=False)

        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False

        # Manually extract stages for intermediate features
        self.feature_blocks = nn.ModuleList([
            nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu),  # 0
            self.model.layer1,  # 1
            self.model.layer2,  # 2
            self.model.layer3,  # 3
        ])

    def forward(self, x):
        # x - B*nt, nf, nx, ny, nz
        x = (x - self.inp_shift)/self.inp_scale
        B,C,X,Y,Z = x.shape
        x = x.reshape(B*C, 1, X, Y, Z)
        features = []
        for i, block in enumerate(self.feature_blocks):
            x = block(x)
            if i in self.layers:
                features.append(x)
        return features
    

class UNetDiscriminatorSN(nn.Module):
    def __init__(self, num_in_ch, num_feat=40, skip_connection=True):
        super().__init__()
        norm = spectral_norm
        self.skip_connection = skip_connection
        self.conv0 = nn.Conv3d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample  
        self.conv1 = norm(nn.Conv3d(num_feat, num_feat * 2, 4, 2, 1, bias=False)) # (w/h + 2 - 4 + 2) / 2 = (w/h) / 2
        self.conv2 = norm(nn.Conv3d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False)) # --> (w/h) / 4
        self.conv3 = norm(nn.Conv3d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False)) # --> (w/h) / 8
        # upsample  
        self.conv4 = norm(nn.Conv3d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False)) # (w/h + 2 - 3 + 1) = w/h
        self.conv5 = norm(nn.Conv3d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv3d(num_feat * 2, num_feat, 3, 1, 1, bias=False))  
        # extra convolutions  
        self.conv7 = norm(nn.Conv3d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv3d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv3d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample  
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample  
        x3 = F.interpolate(x3, scale_factor=2, mode='trilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='trilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='trilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0
        
        # extra convolutions  
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out

