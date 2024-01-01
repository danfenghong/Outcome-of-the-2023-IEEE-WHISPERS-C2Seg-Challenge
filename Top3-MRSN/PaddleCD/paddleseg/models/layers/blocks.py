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
import paddle.nn as nn

__all__ = [
    'BasicConv', 'Conv1x1', 'Conv3x3', 'Conv7x7', 'MaxPool2x2', 'MaxUnPool2x2',
    'ConvTransposed3x3', 'Identity', 'get_norm_layer', 'get_act_layer',
    'make_norm', 'make_act', 'ChannelAttention', 'SpatialAttention','CBAM',
]


def get_norm_layer():
    # TODO: select appropriate norm layer.
    return nn.BatchNorm2D


def get_act_layer():
    # TODO: select appropriate activation layer.
    return nn.ReLU


def make_norm(*args, **kwargs):
    norm_layer = get_norm_layer()
    return norm_layer(*args, **kwargs)


def make_act(*args, **kwargs):
    act_layer = get_act_layer()
    return act_layer(*args, **kwargs)

class ChannelAttention(nn.Layer):
    """
    The channel attention module implementation based on PaddlePaddle.

    The original article refers to 
        Sanghyun Woo, et al., "CBAM: Convolutional Block Attention Module"
        (https://arxiv.org/abs/1807.06521).

    Args:
        in_ch (int): The number of channels of the input features.
        ratio (int, optional): The channel reduction ratio. Default: 8.
    """

    def __init__(self, in_ch, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        self.fc1 = Conv1x1(in_ch, in_ch // ratio, bias=False, act=True)
        self.fc2 = Conv1x1(in_ch // ratio, in_ch, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.fc1(self.avg_pool(x)))
        max_out = self.fc2(self.fc1(self.max_pool(x)))
        out = avg_out + max_out
        return F.sigmoid(out)


class SpatialAttention(nn.Layer):
    """
    The spatial attention module implementation based on PaddlePaddle.

    The original article refers to 
        Sanghyun Woo, et al., "CBAM: Convolutional Block Attention Module"
        (https://arxiv.org/abs/1807.06521).

    Args:
        kernel_size (int, optional): The size of the convolutional kernel. Default: 7.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = BasicConv(2, 1, kernel_size, bias=False)

    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        x = paddle.concat([avg_out, max_out], axis=1)
        x = self.conv(x)
        return F.sigmoid(x)


class CBAM(nn.Layer):
    """
    The CBAM implementation based on PaddlePaddle.

    The original article refers to 
        Sanghyun Woo, et al., "CBAM: Convolutional Block Attention Module"
        (https://arxiv.org/abs/1807.06521).

    Args:
        in_ch (int): The number of channels of the input features.
        ratio (int, optional): The channel reduction ratio for the channel attention module. Default: 8.
        kernel_size (int, optional): The size of the convolutional kernel used in the spatial attention module. Default: 7.
    """

    def __init__(self, in_ch, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_ch, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        y = self.ca(x) * x
        y = self.sa(y) * y
        return y


class BasicConv(nn.Layer):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 pad_mode='constant',
                 bias='auto',
                 norm=False,
                 act=False,
                 **kwargs):
        super(BasicConv, self).__init__()
        seq = []
        if kernel_size >= 2:
            seq.append(nn.Pad2D(kernel_size // 2, mode=pad_mode))
        seq.append(
            nn.Conv2D(
                in_ch,
                out_ch,
                kernel_size,
                stride=1,
                padding=0,
                bias_attr=(False if norm else None) if bias == 'auto' else bias,
                **kwargs))
        if norm:
            if norm is True:
                norm = make_norm(out_ch)
            seq.append(norm)
        if act:
            if act is True:
                act = make_act()
            seq.append(act)
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class Conv1x1(BasicConv):
    def __init__(self,
                 in_ch,
                 out_ch,
                 pad_mode='constant',
                 bias='auto',
                 norm=False,
                 act=False,
                 **kwargs):
        super(Conv1x1, self).__init__(
            in_ch,
            out_ch,
            1,
            pad_mode=pad_mode,
            bias=bias,
            norm=norm,
            act=act,
            **kwargs)


class Conv3x3(BasicConv):
    def __init__(self,
                 in_ch,
                 out_ch,
                 pad_mode='constant',
                 bias='auto',
                 norm=False,
                 act=False,
                 **kwargs):
        super(Conv3x3, self).__init__(
            in_ch,
            out_ch,
            3,
            pad_mode=pad_mode,
            bias=bias,
            norm=norm,
            act=act,
            **kwargs)


class Conv7x7(BasicConv):
    def __init__(self,
                 in_ch,
                 out_ch,
                 pad_mode='constant',
                 bias='auto',
                 norm=False,
                 act=False,
                 **kwargs):
        super(Conv7x7, self).__init__(
            in_ch,
            out_ch,
            7,
            pad_mode=pad_mode,
            bias=bias,
            norm=norm,
            act=act,
            **kwargs)


class MaxPool2x2(nn.MaxPool2D):
    def __init__(self, **kwargs):
        super(MaxPool2x2, self).__init__(kernel_size=2, stride=(2, 2), padding=(0, 0), **kwargs)


class MaxUnPool2x2(nn.MaxUnPool2D):
    def __init__(self, **kwargs):
        super(MaxUnPool2x2, self).__init__(kernel_size=2, stride=(2, 2), padding=(0, 0), **kwargs)


class ConvTransposed3x3(nn.Layer):
    def __init__(self,
                 in_ch,
                 out_ch,
                 bias='auto',
                 norm=False,
                 act=False,
                 **kwargs):
        super(ConvTransposed3x3, self).__init__()
        seq = []
        seq.append(
            nn.Conv2DTranspose(
                in_ch,
                out_ch,
                3,
                stride=2,
                padding=1,
                bias_attr=(False if norm else None) if bias == 'auto' else bias,
                **kwargs))
        if norm:
            if norm is True:
                norm = make_norm(out_ch)
            seq.append(norm)
        if act:
            if act is True:
                act = make_act()
            seq.append(act)
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class Identity(nn.Layer):
    """A placeholder identity operator that accepts exactly one argument."""

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
