# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn), Jingyi Xie (hsfzxjy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

from .backbone.HRNet import HighResolutionNet
from .decoder.FPN_Seg_Decoder import HRNet_FPN_Seg_Decoder

ALIGN_CORNERS = True
BN_MOMENTUM = 0.1

BatchNorm2d=nn.BatchNorm2d

class SegHRNet(nn.Module):
    def __init__(self, in_ch, n_classes, backbone='hr-w32'):
        super(SegHRNet, self).__init__()
        self.backbone = HighResolutionNet(in_ch, backbone=backbone)
        self.decoder = HRNet_FPN_Seg_Decoder(self.backbone.last_inp_channels, n_classes)
    
    def init_weights(self, pretrained=''):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            self.backbone.init_weights(pretrained=pretrained)
        elif pretrained:
            raise RuntimeError('No such file {}'.format(pretrained))

    def forward(self, input):
        x = self.backbone(input)
        x = self.decoder(x)
        output = F.interpolate(x, size=input.shape[2:], mode='bilinear', align_corners=ALIGN_CORNERS) #output = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=ALIGN_CORNERS)
        return output


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = torch.autograd.Variable(torch.randn(1, 3, 512, 512)).to(device)
    net = SegHRNet(3, 17).to(device)
    print(net(input).size())