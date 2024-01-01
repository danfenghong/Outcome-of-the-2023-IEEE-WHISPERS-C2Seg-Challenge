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
class CDNet_Fusion(nn.Layer):
    def __init__(self,
                 architecture,
                 num_classes,
                 backbone,
                 backbone_indices,
                 inut_chs=3,
                 pretrained=None):
        super().__init__()
        if architecture == 'ocrnet':
            self.architecture = OCRNet(num_classes=num_classes,
                                       backbone=backbone,
                                       backbone_indices=backbone_indices,
                                       pretrained=pretrained)

        self.architecture.backbone.conv_layer1_1 = layers.ConvBNReLU(
            in_channels=inut_chs*2,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias_attr=False)


    def forward(self, x1, x2):
        x = paddle.concat([x1,x2],axis=1)

        logit_list = self.architecture(x)
        return logit_list


# @manager.MODELS.add_component
# class CDNet_Fusion1(nn.Layer):
#     def __init__(self,
#                  num_classes,
#                  backbone,
#                  backbone_indices,
#                  ocr_mid_channels=512,
#                  ocr_key_channels=256,
#                  inut_chs=3,
#                  align_corners=False,
#                  pretrained=None):
#         super().__init__()

#         self.backbone = backbone
#         self.backbone.conv_layer1_1 = layers.ConvBNReLU(
#             in_channels=inut_chs*2,
#             out_channels=64,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             bias_attr=False)
#         in_channels = [self.backbone.feat_channels[i] for i in backbone_indices]
#         self.backbone_indices = backbone_indices


#         self.head = OCRHead(
#             num_classes=num_classes,
#             in_channels=in_channels,
#             ocr_mid_channels=ocr_mid_channels,
#             ocr_key_channels=ocr_key_channels)

#         self.align_corners = align_corners
#         self.pretrained = pretrained
#         self.init_weight()

#     def forward(self, x1, x2):
#         print(x1.shape, x2.shape)
#         x = paddle.concat([x1,x2],axis=1)
#         print(x.shape)
#         # print(self.backbone)
#         # print(self.head)
#         feats = self.backbone(x)
#         feats = [feats[i] for i in self.backbone_indices]
#         logit_list = self.head(feats)
#         if not self.training:
#             logit_list = [logit_list[0]]

#         logit_list = [
#             F.interpolate(
#                 logit,
#                 paddle.shape(x)[2:],
#                 mode='bilinear',
#                 align_corners=self.align_corners) for logit in logit_list
#         ]
#         return logit_list

#     def init_weight(self):
#         if self.pretrained is not None:
#             utils.load_entire_model(self, self.pretrained)