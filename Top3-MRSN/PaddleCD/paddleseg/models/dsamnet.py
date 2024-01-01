# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.cvlibs.param_init import KaimingInitMixin
from paddleseg.models.layers.blocks import Conv3x3, Conv1x1, get_norm_layer, Identity, make_norm
from paddleseg.models.layers.attention import CBAM
from paddleseg.models.backbones import resnet


@manager.MODELS.add_component
class DSAMNet(nn.Layer):
    """
    The DSAMNet implementation based on PaddlePaddle.

    The original article refers to
        Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing 
        Change Detection"
        (https://ieeexplore.ieee.org/document/9467555).

    Note that this implementation differs from the original work in two aspects:
    1. We do not use multiple dilation rates in layer 4 of the ResNet backbone.
    2. A classification head is used in place of the original metric learning-based head to stablize the training process.

    Args:
        in_channels (int): The number of bands of the input images.
        num_classes (int): The number of target classes.
        ca_ratio (int, optional): The channel reduction ratio for the channel attention module. Default: 8.
        sa_kernel (int, optional): The size of the convolutional kernel used in the spatial attention module. Default: 7.
    """

    def __init__(self, 
                    in_channels, 
                    num_classes, 
                    backb, 
                    dropout_rate=0.0,
                    ca_ratio=8, 
                    sa_kernel=7):
        super(DSAMNet, self).__init__()

        WIDTH = 64

        self.backbone = Backbone(
            in_ch=in_channels, arch=backb, strides=(1, 1, 2, 2, 1))
        self.decoder = Decoder(WIDTH,dropout_rate=dropout_rate)

        self.cbam1 = CBAM(64, ratio=ca_ratio, kernel_size=sa_kernel)
        self.cbam2 = CBAM(64, ratio=ca_ratio, kernel_size=sa_kernel)

        self.dsl2 = DSLayer(64, num_classes, 32, stride=2, output_padding=1, dropout_rate=dropout_rate)
        self.dsl3 = DSLayer(128, num_classes, 32, stride=4, output_padding=3, dropout_rate=dropout_rate)

        self.conv_out = nn.Sequential(
            Conv3x3(
                WIDTH, WIDTH, norm=True, act=True),
            Conv3x3(WIDTH, num_classes))

        self.init_weight()

    def forward(self, t1, t2):
        f1 = self.backbone(t1)
        f2 = self.backbone(t2)

        y1 = self.decoder(f1)
        y2 = self.decoder(f2)

        y1 = self.cbam1(y1)
        y2 = self.cbam2(y2)

        out = paddle.abs(y1 - y2)
        out = F.interpolate(
            out, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)
        pred = self.conv_out(out)

        if not self.training:
            return [pred]
        else:
            ds2 = self.dsl2(paddle.abs(f1[0] - f2[0]))
            ds3 = self.dsl3(paddle.abs(f1[1] - f2[1]))
            return [pred, ds2, ds3]

    def init_weight(self):
        pass


class DSLayer(nn.Sequential):
    def __init__(self, in_ch, out_ch, itm_ch, dropout_rate=0,**convd_kwargs):
        super(DSLayer, self).__init__(
            nn.Conv2DTranspose(
                in_ch, itm_ch, kernel_size=3, padding=1, **convd_kwargs),
            make_norm(itm_ch),
            nn.ReLU(),
            nn.Dropout2D(p=dropout_rate),
            nn.Conv2DTranspose(
                itm_ch, out_ch, kernel_size=3, padding=1))


class Backbone(nn.Layer, KaimingInitMixin):
    def __init__(self, in_ch, arch, pretrained=True, strides=(2, 1, 2, 2, 2)):
        super(Backbone, self).__init__()

        if arch == 'resnet18':
            self.resnet = resnet.resnet18(
                pretrained=pretrained,
                strides=strides,
                norm_layer=get_norm_layer())
        elif arch == 'resnet34':
            self.resnet = resnet.resnet34(
                pretrained=pretrained,
                strides=strides,
                norm_layer=get_norm_layer())
        elif arch == 'resnet50':
            self.resnet = resnet.resnet50(
                pretrained=pretrained,
                strides=strides,
                norm_layer=get_norm_layer())
        else:
            raise ValueError

        self._trim_resnet()

        if in_ch != 3:
            self.resnet.conv1 = nn.Conv2D(
                in_ch,
                64,
                kernel_size=7,
                stride=strides[0],
                padding=3,
                bias_attr=False)

        if not pretrained:
            self.init_weight()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        return x1, x2, x3, x4

    def _trim_resnet(self):
        self.resnet.avgpool = Identity()
        self.resnet.fc = Identity()


class Decoder(nn.Layer, KaimingInitMixin):
    def __init__(self, f_ch, dropout_rate=0):
        super(Decoder, self).__init__()
        self.dr1 = Conv1x1(64, 96, norm=True, act=True)
        self.dr2 = Conv1x1(128, 96, norm=True, act=True)
        self.dr3 = Conv1x1(256, 96, norm=True, act=True)
        self.dr4 = Conv1x1(512, 96, norm=True, act=True)
        self.conv_out = nn.Sequential(
            Conv3x3(
                384, 256, norm=True, act=True),
            nn.Dropout(dropout_rate),
            Conv1x1(
                256, f_ch, norm=True, act=True))

        self.init_weight()

    def forward(self, feats):
        f1 = self.dr1(feats[0])
        f2 = self.dr2(feats[1])
        f3 = self.dr3(feats[2])
        f4 = self.dr4(feats[3])

        f2 = F.interpolate(
            f2, size=paddle.shape(f1)[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(
            f3, size=paddle.shape(f1)[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(
            f4, size=paddle.shape(f1)[2:], mode='bilinear', align_corners=True)

        x = paddle.concat([f1, f2, f3, f4], axis=1)
        y = self.conv_out(x)

        return y