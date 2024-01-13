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

import paddle.nn as nn


def constant_init(param, **kwargs):
    """
    Initialize the `param` with constants.

    Args:
        param (Tensor): Tensor that needs to be initialized.

    Examples:

        from paddleseg.cvlibs import param_init
        import paddle.nn as nn

        linear = nn.Linear(2, 4)
        param_init.constant_init(linear.weight, value=2.0)
        print(linear.weight.numpy())
        # result is [[2. 2. 2. 2.], [2. 2. 2. 2.]]

    """
    initializer = nn.initializer.Constant(**kwargs)
    initializer(param, param.block)


def normal_init(param, **kwargs):
    """
    Initialize the `param` with a Normal distribution.

    Args:
        param (Tensor): Tensor that needs to be initialized.

    Examples:

        from paddleseg.cvlibs import param_init
        import paddle.nn as nn

        linear = nn.Linear(2, 4)
        param_init.normal_init(linear.weight, loc=0.0, scale=1.0)

    """
    initializer = nn.initializer.Normal(**kwargs)
    initializer(param, param.block)


def kaiming_normal_init(param, **kwargs):
    r"""
    Initialize the input tensor with Kaiming Normal initialization.

    This function implements the `param` initialization from the paper
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification <https://arxiv.org/abs/1502.01852>`
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. This is a
    robust initialization method that particularly considers the rectifier
    nonlinearities. In case of Uniform distribution, the range is [-x, x], where
    .. math::
        x = \sqrt{\\frac{6.0}{fan\_in}}
    In case of Normal distribution, the mean is 0 and the standard deviation
    is
    .. math::
        \sqrt{\\frac{2.0}{fan\_in}}

    Args:
        param (Tensor): Tensor that needs to be initialized.

    Examples:

        from paddleseg.cvlibs import param_init
        import paddle.nn as nn

        linear = nn.Linear(2, 4)
        # uniform is used to decide whether to use uniform or normal distribution
        param_init.kaiming_normal_init(linear.weight)

    """
    initializer = nn.initializer.KaimingNormal(**kwargs)
    initializer(param, param.block)



class KaimingInitMixin:
    """
    A mix-in that provides the Kaiming initialization functionality.

    Examples:

        from paddlers.custom_models.cd.models.param_init import KaimingInitMixin

        class CustomNet(nn.Layer, KaimingInitMixin):
            def __init__(self, num_channels, num_classes):
                super().__init__()
                self.conv = nn.Conv2D(num_channels, num_classes, 3, 1, 0, bias_attr=False)
                self.bn = nn.BatchNorm2D(num_classes)
                self.init_weight()
    """

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                kaiming_normal_init(layer.weight)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                constant_init(layer.weight, value=1)
                constant_init(layer.bias, value=0)



                