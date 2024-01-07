from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

ALIGN_CORNERS = True
BN_MOMENTUM = 0.1

BatchNorm2d=nn.BatchNorm2d

# class Vanilla_FPN_Decoder(nn.Module):
#     def __init__(self, filters):
#         super(Vanilla_FPN_Decoder, self).__init__()
#         self.reduce1 = nn.Conv2d(filters[3], filters[3], 1, 1, 0)
#         self.reduce2 = nn.Conv2d(filters[3], filters[2], 1, 1, 0)
#         self.reduce3 = nn.Conv2d(filters[2], filters[1], 1, 1, 0)
#         self.reduce4 = nn.Conv2d(filters[1], filters[0], 1, 1, 0)
    
#     def forward(self, x):
#         print(x[3].shape,x[2].shape,x[1].shape,x[0].shape)
#         """
#         torch.Size([32, 384, 4, 4]) torch.Size([32, 192, 8, 8]) 
#         torch.Size([32, 96, 16, 16]) torch.Size([32, 48, 32, 32])
#         """
#         d4 = self.reduce1(x[3]) + x[3]
#         d3 = self.reduce2(d4) + x[2]
#         d2 = self.reduce3(d3) + x[1]
#         d1 = self.reduce4(d2) + x[0]
#         return d1

class Vanilla_FPN_Decoder(nn.Module):
    def __init__(self, filters, n_classes, dim=256):
        super(Vanilla_FPN_Decoder, self).__init__()
        self.reduce1 = nn.Conv2d(filters[3], dim, 1, 1, 0)
        self.reduce2 = nn.Conv2d(filters[2], dim, 1, 1, 0)
        self.reduce3 = nn.Conv2d(filters[1], dim, 1, 1, 0)
        self.reduce4 = nn.Conv2d(filters[0], dim, 1, 1, 0)
        # self.last = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=dim,
        #         out_channels=dim,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0),
        #     BatchNorm2d(dim, momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(
        #         in_channels=dim,
        #         out_channels=n_classes,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0)
        # )
    
    def forward(self, x):
        out = self.reduce1(x[3])
        out = self.reduce2(x[2]) + F.interpolate(out, size=(x[2].size(-2), x[2].size(-1)), mode='bilinear', align_corners=True)
        out = self.reduce3(x[1]) + F.interpolate(out, size=(x[1].size(-2), x[1].size(-1)), mode='bilinear', align_corners=True)
        out = self.reduce4(x[0]) + F.interpolate(out, size=(x[0].size(-2), x[0].size(-1)), mode='bilinear', align_corners=True)
        # out = self.last(out)
        return out

class HRNet_FPN_Seg_OCR_Decoder(nn.Module):

    def __init__(self):
        super(HRNet_FPN_Seg_OCR_Decoder, self).__init__()
    
    def forward(self, x):
        assert len(x) == 4
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)

        feats = torch.cat([x[0], x1, x2, x3], 1)
        return feats

class HRNet_FPN_Seg_Decoder(nn.Module):

    def __init__(self, last_inp_channels, n_classes):
        super(HRNet_FPN_Seg_Decoder, self).__init__()
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=n_classes,
                kernel_size=1,
                stride=1,
                padding=0)
        )
    
    def forward(self, x):
        assert len(x) == 4
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)

        feats = torch.cat([x[0], x1, x2, x3], 1)
        out = self.last_layer(feats)
        return out