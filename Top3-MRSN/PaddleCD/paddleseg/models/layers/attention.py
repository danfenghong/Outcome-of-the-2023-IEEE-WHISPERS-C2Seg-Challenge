# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddleseg.models import layers


class AttentionBlock(nn.Layer):
    """General self-attention block/non-local block.

    The original article refers to refer to https://arxiv.org/abs/1706.03762.
    Args:
        key_in_channels (int): Input channels of key feature.
        query_in_channels (int): Input channels of query feature.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        share_key_query (bool): Whether share projection weight between key
            and query projection.
        query_downsample (nn.Module): Query downsample module.
        key_downsample (nn.Module): Key downsample module.
        key_query_num_convs (int): Number of convs for key/query projection.
        value_out_num_convs (int): Number of convs for value projection.
        key_query_norm (bool): Whether to use BN for key/query projection.
        value_out_norm (bool): Whether to use BN for value projection.
        matmul_norm (bool): Whether normalize attention map with sqrt of
            channels
        with_out (bool): Whether use out projection.
    """

    def __init__(self, key_in_channels, query_in_channels, channels,
                 out_channels, share_key_query, query_downsample,
                 key_downsample, key_query_num_convs, value_out_num_convs,
                 key_query_norm, value_out_norm, matmul_norm, with_out):
        super(AttentionBlock, self).__init__()
        if share_key_query:
            assert key_in_channels == query_in_channels
        self.with_out = with_out
        self.key_in_channels = key_in_channels
        self.query_in_channels = query_in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.share_key_query = share_key_query
        self.key_project = self.build_project(
            key_in_channels,
            channels,
            num_convs=key_query_num_convs,
            use_conv_module=key_query_norm)
        if share_key_query:
            self.query_project = self.key_project
        else:
            self.query_project = self.build_project(
                query_in_channels,
                channels,
                num_convs=key_query_num_convs,
                use_conv_module=key_query_norm)

        self.value_project = self.build_project(
            key_in_channels,
            channels if self.with_out else out_channels,
            num_convs=value_out_num_convs,
            use_conv_module=value_out_norm)

        if self.with_out:
            self.out_project = self.build_project(
                channels,
                out_channels,
                num_convs=value_out_num_convs,
                use_conv_module=value_out_norm)
        else:
            self.out_project = None

        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm

    def build_project(self, in_channels, channels, num_convs, use_conv_module):
        if use_conv_module:
            convs = [
                layers.ConvBNReLU(
                    in_channels=in_channels,
                    out_channels=channels,
                    kernel_size=1,
                    bias_attr=False)
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    layers.ConvBNReLU(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=1,
                        bias_attr=False))
        else:
            convs = [nn.Conv2D(in_channels, channels, 1)]
            for _ in range(num_convs - 1):
                convs.append(nn.Conv2D(channels, channels, 1))

        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, query_feats, key_feats):
        query_shape = paddle.shape(query_feats)
        query = self.query_project(query_feats)
        if self.query_downsample is not None:
            query = self.query_downsample(query)
        query = query.flatten(2).transpose([0, 2, 1])

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)

        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)

        key = key.flatten(2)
        value = value.flatten(2).transpose([0, 2, 1])
        sim_map = paddle.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.channels**-0.5) * sim_map
        sim_map = F.softmax(sim_map, axis=-1)

        context = paddle.matmul(sim_map, value)
        context = paddle.transpose(context, [0, 2, 1])

        context = paddle.reshape(
            context, [0, self.out_channels, query_shape[2], query_shape[3]])

        if self.out_project is not None:
            context = self.out_project(context)
        return context

class DR(nn.Layer):
    def __init__(self, in_d, out_d):
        super(DR, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv1 = nn.Conv2D(self.in_d, self.out_d, 1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(self.out_d)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Layer):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)

        self.fc1 = nn.Conv2D(in_planes, in_planes // ratio, 1, bias_attr=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2D(in_planes // ratio, in_planes, 1, bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2D(2, 1, kernel_size, padding=padding, bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        x = paddle.concat([avg_out, max_out], axis=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Layer):
    def __init__(self, in_planes, ratio, kernel_size):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class DS_layer(nn.Layer):
    def __init__(self, in_d, out_d, stride, output_padding, n_class):
        super(DS_layer, self).__init__()

        self.dsconv = nn.Conv2DTranspose(in_d, out_d, kernel_size=3, padding=1, stride=stride,
                                         output_padding=output_padding)
        self.bn = nn.BatchNorm2D(out_d)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2D(p=0.2)
        self.outconv = nn.Conv2DTranspose(out_d, n_class, kernel_size=3, padding=1)

    def forward(self, input):
        x = self.dsconv(input)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.outconv(x)
        return x
