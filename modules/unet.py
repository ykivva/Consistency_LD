import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint

from models import TrainableModel
from utils import *


class UNet_up_block(nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel, up_sample=True):
        super().__init__()
        self.up_sampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, output_channel)        
        self.relu = torch.nn.ReLU()
        self.up_sample = up_sample

    def forward(self, prev_feature_map, x):
        if self.up_sample:
            x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x
    
    def save(self, )


class UNet_down_block(nn.Module):
    def __init__(self, input_channel, output_channel, down_size=True):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, output_channel)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        if self.down_size:
            x = self.max_pool(x)
        return x
    

class Unet_LS_down(nn.Module):
    def __init__(self, downsample=6, in_channel=3):
        super().__init__()
        
        self.downsample, self.in_channel = downsample, in_channel
        self.down1 = UNet_down_block(in_channels, 16, False)
        self.down_blocks = nn.ModuleList(
            [UNet_down_block(2**(4+i), 2**(5+i), True) for i in range(0, downsample)]
        )

        bottleneck = 2**(4 + downsample)
        self.mid_conv1 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, bottleneck)
        self.mid_conv2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, bottleneck)
        
        self.xvals = []

    def forward(self, x):
        x = self.down1(x)
        self.xvals = [x]
        for i in range(0, self.downsample):
            x = self.down_blocks[i](x)
            self.xvals.append(x)

        x = self.relu(self.bn1(self.mid_conv1(x)))
        x = self.relu(self.bn2(self.mid_conv2(x)))
        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

        
class Unet_LS_up(nn.Module):
    def __init__(self, downsample=6, out_channel=3):
        super().__init__()
        
        self.downsample, self.out_channel = downsample, out_channel
        
        bottleneck = 2**(4 + downsample)
        self.mid_conv3 = torch.nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, bottleneck)
        self.mid_conv4 = torch.nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn4 = nn.GroupNorm(8, bottleneck)

        self.up_blocks = nn.ModuleList(
            [UNet_up_block(2**(4+i), 2**(5+i), 2**(4+i)) for i in range(0, downsample)]
        )

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 16)
        self.last_conv2 = nn.Conv2d(16, out_channels, 1, padding=0)
        self.relu = nn.ReLU()
    
    def forward(self, xvals, x):
        x = self.relu(self.bn3(self.mid_conv3(x)))
        x = self.relu(self.bn4(self.mid_conv4(x)))

        for i in range(0, self.downsample)[::-1]:
            x = self.up_blocks[i](xvals[i], x)

        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.relu(self.last_conv2(x))
        
        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

        
class UNet_LS(TrainableModel):
    def __init__(self,  downsample=6, in_channel=3, out_channel=3,
                 model_down=None, model_up=None,
                 path_down=None, path_up=None):
        super().__init__()

        self.in_channel, self.out_channel, self.downsample = in_channel, out_channel, downsample
        
        if not isinstance(model_down, UNet_LS_down):
            model_down = UNet_LS_down(downsample=self.downsample, in_channel=self.in_channel)
        else:
            self.in_channel = model_down.in_channel
            self.downsample = model_down.downsample
        if not isinstance(model_up, UNet_LS_up):
            model_up = UNet_LS_up(downsampel=self.downsample, out_channel=self.out_channel)
        else:
            self.out_channel = model_up.out_channel
            self.downsample = model_up.downsample
        
        assert model_down.downsample==model_up.downsample, "UNet up-model is not match UNet down-model"
        self.blocks = nn.ModuleList([model_down, model_up])
        
        self.load_weights(path_down=path_down, path_up=path_up)

    def forward(self, x):
        x = self.blocks[0](x)
        x = self.blocks[1](self.blocks[0].xvals, x)
        return x
    
    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)
    
    def save(self, path, path_down=None, path_up=None, separated=True, together=False):
        dict_together = {'downsample': self.downsample, 
                         'in_channel': self.in_channel,
                         'out_channel': self.out_channel}
        if together:
            dict_together['model_down_state_dict'] = self.blocks[0].state_dict()
            dict_together['model_up_state_dict'] = self.blocks[1].state_dict()
        
        torch.save(dict_together, path)
        
        if (not together) or separated:
            assert path_down==None, "You should specify path to the Unet down-block"
            assert path_up==None, "You should specify path to the Unet up-block"
            self.blocks[0].save(path_down)
            self.blocks[1].save(path_up)
    
    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path)
        downsample = checkpoint['downsample']
        in_channel = checkpoint['in_channel']
        out_channel = checkpoint['out_channel']
        
        model_up = UNet_LS_up(downsample=downsample, out_channel=out_channel)
        model_down = UNet_LS_down(downsample=downsample, in_channle=in_channel)
        model_up.load_state_dict(checkpoint['model_up_state_dict'])
        model_down.load_state_dict(checkpoint['model_down_state_dict'])
        
        model = cls(downsample=downsample, in_channel=in_channel, out_channel=out_channel,
                    model_down=model_down, model_up=model_up)
        return model
        
    def load_weights(self, path={}):
        path_up = path.get("up", None)
        path_down = path.get("down", None)
        if path_down is not None:
            self.block[0].load_weights(path_down)
        
        if path_up is not None:
            self.block[1].load_weights(path_up)
            