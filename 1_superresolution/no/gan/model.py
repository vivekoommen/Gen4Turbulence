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

class FeatureExtractor(nn.Module):
    def __init__(self, layers=[0, 5, 10, 19, 28]):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.layers = layers
        self.net = nn.Sequential(*[vgg19_model.features[i] for i in range(max(layers) + 1)])

    def forward(self, img):
        # img - [B, 1, X, Y]
        img = torch.tile(img, (1,3,1,1))
        features = []
        for i in range(len(self.net)):
            img = self.net[i](img)
            if i in self.layers:
                features.append(img)
        return features
    

class UNetDiscriminatorSN(nn.Module):
    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super().__init__()
        norm = spectral_norm
        self.skip_connection = skip_connection
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample  
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False)) # (w/h + 2 - 4 + 2) / 2 = (w/h) / 2
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False)) # --> (w/h) / 4
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False)) # --> (w/h) / 8
        # upsample  
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False)) # (w/h + 2 - 3 + 1) = w/h
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))  
        # extra convolutions  
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample  
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample  
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0
        
        # extra convolutions  
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


class DenseResidualBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        '''
        filters = input channels 
        '''
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)
        
        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x
    

class GeneratorRRDB(nn.Module):
    def __init__(self, channels, Par, filters=32, num_res_blocks=4, num_upsample=1):
        super(GeneratorRRDB, self).__init__()

        self.Par=Par

        # First Layer  
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual block  
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Upsampling layers  
        # upsample_layers = []
        # for _ in range(num_upsample):
        #     upsample_layers += [
        #         nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1), 
        #         nn.LeakyReLU(), 
        #         nn.PixelShuffle(upscale_factor=2), 
        #     ]
        # self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block  
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(), 
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1), 
        )

    def forward(self, x):
        x = (x - self.Par["inp_shift"])/self.Par["inp_scale"]
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        # out = self.upsampling(out)
        out = self.conv3(out)
        out = out*self.Par["out_scale"] + self.Par["out_shift"]
        return out