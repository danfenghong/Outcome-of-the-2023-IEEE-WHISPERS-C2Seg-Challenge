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
from .decoder.FPN_Seg_Decoder import Vanilla_FPN_Decoder

ALIGN_CORNERS = True
BN_MOMENTUM = 0.1

BatchNorm2d=nn.BatchNorm2d

class ClassifierModule(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding,
                          dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out

class SegHRNet_DA(nn.Module):
    def __init__(self, in_ch, n_classes, backbone='hr-w32'):
        super(SegHRNet_DA, self).__init__()
        self.backbone = HighResolutionNet(in_ch, backbone=backbone)
        if in_ch == 248:
            self.in_ch_list = [4,2,242]
        elif in_ch == 122:
            self.in_ch_list = [4,2,116]
        self.backbone1 = HighResolutionNet(self.in_ch_list[0], backbone=backbone)
        self.backbone2 = HighResolutionNet(self.in_ch_list[1], backbone=backbone)
        self.backbone3 = HighResolutionNet(self.in_ch_list[2], backbone=backbone)
        filters = self.backbone.get_filters()
        self.decoder = Vanilla_FPN_Decoder(filters, n_classes, dim=64)
        self.clf1 = ClassifierModule(filters[3], dilation_series=[8,4,2,1], padding_series=[8,4,2,1], num_classes=n_classes)
        self.clf2 = ClassifierModule(64, dilation_series=[8,4,2,1], padding_series=[8,4,2,1], num_classes=n_classes)

    def init_weights(self, pretrained=''):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            # self.backbone.init_weights(pretrained=pretrained)
            # self.backbone1.init_weights(pretrained=pretrained)
            # self.backbone2.init_weights(pretrained=pretrained)
            # self.backbone3.init_weights(pretrained=pretrained)
            self.backbone.init_weights(pretrained="/Top2-Seg-HRN/run/train/xxx/best_model.pt")
            self.backbone1.init_weights(pretrained="/Top2-Seg-HRN/run/train/xxx/best_model.pt")
            self.backbone2.init_weights(pretrained="/Top2-Seg-HRN/run/train/xxx/best_model.pt")
            self.backbone3.init_weights(pretrained="/Top2-Seg-HRN/run/train/xxx/best_model.pt")
        elif pretrained:
            raise RuntimeError('No such file {}'.format(pretrained))

    def forward(self, input):
        x = self.backbone(input)
        x1 = self.backbone1(input[:,:self.in_ch_list[0],:,:])
        x2 = self.backbone2(input[:,:self.in_ch_list[1],:,:])
        x3 = self.backbone3(input[:,:self.in_ch_list[2],:,:])
        x = x+x1+x2+x3
        out = self.decoder(x)
        x1 = self.clf1(x[3])
        out1 = self.clf2(out)
        return out1, x1

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = torch.autograd.Variable(torch.randn(1, 3, 512, 512)).to(device)
    net = SegHRNet_DA(3, 17).to(device)
    x1, x2 = net(input)
    print(x1.size())
    print(x2.size())

