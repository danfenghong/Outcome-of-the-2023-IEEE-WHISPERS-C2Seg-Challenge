# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

# Code was based on https://github.com/facebookresearch/ConvNeXt
# from https://github.com/PaddlePaddle/PASSL/blob/main/passl/modeling/backbones/convnext.py

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from collections import OrderedDict


from paddle.utils.download import get_weights_path_from_url

__all__ = []

model_urls = {
    'convnext_tiny': ('https://passl.bj.bcebos.com/models/convnext_tiny_1k_224.pdparams',
                 'c5541588b0906df1b853982e1d463eec'),
    'convnext_small': ('https://passl.bj.bcebos.com/models/convnext_small_1k_224.pdparams',
                 'f4a6f26284889fa0953a2fe5b8167215'),
    'convnext_base':('data/data127804/convnext_base_1k_224_ema.pdparams', '')
}

trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)
zeros_ = nn.initializer.Constant(value=0.0)
ones_ = nn.initializer.Constant(value=1.0)


class Identity(nn.Layer):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Layer):
    """ ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2D(dim, dim, kernel_size=7, padding=3,
                                groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, epsilon=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(
                value=layer_scale_init_value)
        ) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.transpose([0, 2, 3, 1])  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose([0, 3, 1, 2])  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Layer):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self,
                 normalized_shape,
                 epsilon=1e-6,
                 data_format="channels_last"):
        super().__init__()

        self.weight = paddle.create_parameter(shape=[normalized_shape],
                                              dtype='float32',
                                              default_initializer=ones_)

        self.bias = paddle.create_parameter(shape=[normalized_shape],
                                            dtype='float32',
                                            default_initializer=zeros_)

        self.epsilon = epsilon
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.epsilon)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / paddle.sqrt(s + self.epsilon)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class ConvNeXt(nn.Layer):
    """ ConvNeXt
        A Paddle impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        strides=(4, 2, 2, 2),
        drop_path_rate=0.,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()
        self.dims = dims
        self.downsample_layers = nn.LayerList(
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2D(in_chans, dims[0], kernel_size=strides[0], stride=strides[0]),
            LayerNorm(dims[0], epsilon=1e-6, data_format="channels_first"),
            nn.Upsample(scale_factor=2))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], epsilon=1e-6, data_format="channels_first"),
                nn.Conv2D(dims[i], dims[i + 1], kernel_size=2, stride=strides[i+1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.LayerList(
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [
            x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))
        ]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[
                Block(dim=dims[i],
                      drop_path=dp_rates[cur + j],
                      layer_scale_init_value=layer_scale_init_value)
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], epsilon=1e-6)  # final norm layer

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2D, nn.Linear)):
            trunc_normal_(m.weight)
            zeros_(m.bias)

    def forward(self, x):
        outputs = []
        for i in range(3):
            # print(x.shape)
            # print(self.downsample_layers[i])
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outputs.append(x)
        return outputs
        # return self.norm(x.mean([-2, -1]))

def _convnext(arch, depths, pretrained, **kwargs):
    model = ConvNeXt(depths=depths, **kwargs)
    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        if arch != 'convnext_base':
            weight_path = get_weights_path_from_url(model_urls[arch][0], model_urls[arch][1])
        else:
            weight_path = model_urls[arch][0]

        param = paddle.load(weight_path)
        param = keys_process(param)
        # print(list(param.keys())[:10])
        # print(list(model.state_dict().keys())[:10])
        model.set_dict(param)
        print('model load from',weight_path)
    return model

def keys_process(param):
    param_remap = OrderedDict()
    for k, v in param.items():
        k_remap = k.replace('backbone.', '')
        param_remap[k_remap] = v
    return param_remap

def convnext_tiny(pretrained=True, **kwargs):

    return _convnext('convnext_tiny',  depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], pretrained=pretrained, **kwargs)


def convnext_small(pretrained=True, **kwargs):

    return _convnext('convnext_small', depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], pretrained=pretrained, **kwargs)

def convnext_base(pretrained=True, in_22k=False, **kwargs):

    return _convnext('convnext_base', depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], pretrained=pretrained, **kwargs)


def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model


def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    return model