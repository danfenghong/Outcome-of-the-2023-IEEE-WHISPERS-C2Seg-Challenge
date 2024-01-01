# -*- coding: utf-8 -*- 
# @File             : cdnet_fusion.py 
# @Author           : zhaoHL
# @Contact          : huilin16@qq.com
# @Time Create First: 2021/9/11 13:18 
# @Contributor      : zhaoHL
# @Time Modify Last : 2021/9/11 13:18 
'''
@File Description:

'''
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers
from paddleseg.models import *

@manager.MODELS.add_component
class CDNet_Parallel(nn.Layer):
    def __init__(self,
                 backbone,
                 head_bin,
                 head=None,
                 num_classes=None,
                 inut_chs=3,
                 lightweight=False,
                 pretrained=None):
        super().__init__()

        self.backbone = backbone

        # if
        in_channels = np.sum(self.backbone.stage4_num_channels)

        if head=='fcn':
            self.head = FCNHead(in_channels, 1, lightweight)
        elif head=='psp':
            self.head = PSPHead(in_channels, 1, lightweight)
        else:
            self.head = None

        if head_bin=='fcn':
            self.head_bin = None #FCNHead(in_channels, 1, lightweight)
        elif head_bin=='psp':
            self.head_bin = PSPHead(in_channels, 1, lightweight)
        else:
            self.head_bin = None

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x1, x2, tta=False):
        if not tta:
            b, c, h, w = x1.shape
            x1 = self.backbone(x1)[-1]
            x2 = self.backbone(x2)[-1]


            out_bin = paddle.abs(x1 - x2)
            out_bin = self.head_bin(out_bin)
            out_bin = F.interpolate(out_bin, size=(h, w), mode='bilinear', align_corners=False)
            out_bin = paddle.nn.functional.sigmoid(out_bin)

            if self.head is not None:
                out1 = self.head(x1)
                out2 = self.head(x2)

                out1 = F.interpolate(out1, size=(h, w), mode='bilinear', align_corners=False)
                out2 = F.interpolate(out2, size=(h, w), mode='bilinear', align_corners=False)

                return out1, out2, out_bin.squeeze(1)
            else:
                return [out_bin.squeeze(1)]
        else:
            return self.tta_forward(x1, x2)

    def tta_forward(self, x1, x2):
        out1, out2, out_bin = self.base_forward(x1, x2)
        out1 = F.softmax(out1, axis=1)
        out2 = F.softmax(out2, axis=1)
        out_bin = out_bin.unsqueeze(1)
        origin_x1 = x1.clone()
        origin_x2 = x2.clone()

        x1 = origin_x1.flip(2)
        x2 = origin_x2.flip(2)
        cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
        out1 += F.softmax(cur_out1, axis=1).flip(2)
        out2 += F.softmax(cur_out2, axis=1).flip(2)
        out_bin += cur_out_bin.unsqueeze(1).flip(2)

        x1 = origin_x1.flip(3)
        x2 = origin_x2.flip(3)
        cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
        out1 += F.softmax(cur_out1, axis=1).flip(3)
        out2 += F.softmax(cur_out2, axis=1).flip(3)
        out_bin += cur_out_bin.unsqueeze(1).flip(3)

        x1 = origin_x1.transpose(2, 3).flip(3)
        x2 = origin_x2.transpose(2, 3).flip(3)
        cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
        out1 += F.softmax(cur_out1, axis=1).flip(3).transpose(2, 3)
        out2 += F.softmax(cur_out2, axis=1).flip(3).transpose(2, 3)
        out_bin += cur_out_bin.unsqueeze(1).flip(3).transpose(2, 3)

        x1 = origin_x1.flip(3).transpose(2, 3)
        x2 = origin_x2.flip(3).transpose(2, 3)
        cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
        out1 += F.softmax(cur_out1, axis=1).transpose(2, 3).flip(3)
        out2 += F.softmax(cur_out2, axis=1).transpose(2, 3).flip(3)
        out_bin += cur_out_bin.unsqueeze(1).transpose(2, 3).flip(3)

        x1 = origin_x1.flip(2).flip(3)
        x2 = origin_x2.flip(2).flip(3)
        cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
        out1 += F.softmax(cur_out1, axis=1).flip(3).flip(2)
        out2 += F.softmax(cur_out2, axis=1).flip(3).flip(2)
        out_bin += cur_out_bin.unsqueeze(1).flip(3).flip(2)

        out1 /= 6.0
        out2 /= 6.0
        out_bin /= 6.0

        return out1, out2, out_bin.squeeze(1)

class PSPHead(nn.Layer):
    def __init__(self, in_channels, out_channels, lightweight):
        super(PSPHead, self).__init__()
        inter_channels = in_channels // 4

        self.conv5 = nn.Sequential(PyramidPooling(in_channels),
                                   conv3x3(in_channels + 4 * int(in_channels / 4), inter_channels, lightweight),
                                   nn.BatchNorm2D(inter_channels),
                                   nn.ReLU(True),
                                   nn.Dropout(0.1, False),
                                   nn.Conv2D(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)


class PyramidPooling(nn.Layer):
    def __init__(self, in_channels):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2D(1)
        self.pool2 = nn.AdaptiveAvgPool2D(2)
        self.pool3 = nn.AdaptiveAvgPool2D(3)
        self.pool4 = nn.AdaptiveAvgPool2D(6)

        out_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
                                   nn.BatchNorm2D(out_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
                                   nn.BatchNorm2D(out_channels),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
                                   nn.BatchNorm2D(out_channels),
                                   nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
                                   nn.BatchNorm2D(out_channels),
                                   nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        f1 = self.conv1(self.pool1(x))
        feat1 = F.interpolate(f1, (h, w), mode="bilinear", align_corners=False)
        f2 = self.conv2(self.pool2(x))
        feat2 = F.interpolate(f2, (h, w), mode="bilinear", align_corners=False)
        f3 = self.conv3(self.pool3(x))
        feat3 = F.interpolate(f3, (h, w), mode="bilinear", align_corners=False)
        f4 = self.conv4(self.pool4(x))
        feat4 = F.interpolate(f4, (h, w), mode="bilinear", align_corners=False)
        return paddle.concat((x, feat1, feat2, feat3, feat4), 1)


class FCNHead(nn.Layer):
    def __init__(self, in_channels, out_channels, lightweight):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4

        self.head = nn.Sequential(conv3x3(in_channels, inter_channels, lightweight),
                                  nn.BatchNorm2D(inter_channels),
                                  nn.ReLU(True),
                                  nn.Dropout(0.1, False),
                                  nn.Conv2D(inter_channels, out_channels, 1, bias_attr=True))

    def forward(self, x):
        return self.head(x)


class DSConv(nn.Layer):
    def __init__(self, in_channels, out_channels, atrous_rate=1):
        super(DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(in_channels, in_channels, 3, padding=atrous_rate, groups=in_channels,
                      dilation=atrous_rate, bias_attr=False),
            nn.BatchNorm2D(in_channels),
            nn.ReLU(True),
            nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
        )

    def forward(self, x):
        return self.conv(x)

def conv3x3(in_channels, out_channels, lightweight, atrous_rate=1):
    if lightweight:
        return DSConv(in_channels, out_channels, atrous_rate)
    else:
        return nn.Conv2D(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias_attr=False)
