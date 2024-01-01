import paddle
from paddle import nn, Tensor
from paddle.nn import functional as F
from typing import Tuple


from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.cvlibs.param_init import KaimingInitMixin
from paddleseg.models.layers.blocks import Conv3x3, Conv1x1, get_norm_layer, Identity, make_norm
from paddleseg.models.layers.attention import CBAM
from paddleseg.models.backbones import resnet, convnext



@manager.MODELS.add_component
class CX_Uper_4B(nn.Layer):
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
                    hsi_chs=242, 
                    dropout_rate=0.0,
                    ):
        super(CX_Uper_4B, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone0 = convnext.convnext_tiny()
            self.backbone1 = convnext.convnext_tiny()
            self.backbone2 = convnext.convnext_tiny(in_chans=2)
            self.backbone3 = convnext.convnext_tiny(in_chans=hsi_chs)
        elif backb == 'convnext_small':
            self.backbone0 = convnext.convnext_small()
            self.backbone1 = convnext.convnext_small()
            self.backbone2 = convnext.convnext_small(in_chans=2)
            self.backbone3 = convnext.convnext_small(in_chans=hsi_chs)
        else:
            self.backbone0 = convnext.convnext_base()
            self.backbone1 = convnext.convnext_base()
            self.backbone2 = convnext.convnext_base(in_chans=2)
            self.backbone3 = convnext.convnext_base(in_chans=hsi_chs)
        

        self.decode_head2b = UPerHead_3B(self.backbone1.dims[:3], num_classes=num_classes)
        self.decode_head3b = UPerHead(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

    def forward(self, t1, t2):
        t3 = t2
        t0 = paddle.concat([t1[:, :2, ...], t1[:, 3:4, ...]], axis=1)
        t2 = t1[:, 4:, ...]
        t1 = t1[:, :3, ...]
        fs0 = self.backbone0(t0)
        fs1 = self.backbone1(t1)
        fs2 = self.backbone2(t2)
        fs3 = self.backbone3(t3)
        fs_diff = []
        for f0, f1, f2 in zip(fs0, fs1, fs2):
            f = self.drop(paddle.concat([f0, f1, f2], axis=1))
            fs_diff.append(f)
        y2 = self.decode_head2b(fs_diff)
        y3 = self.decode_head3b(fs3)
        y = y2+y3
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]


@manager.MODELS.add_component
class CX_Uper_2B(nn.Layer):
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
                    ):
        super(CX_Uper_2B, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone1 = convnext.convnext_tiny()
            self.backbone2 = convnext.convnext_tiny()
        elif backb == 'convnext_small':
            self.backbone1 = convnext.convnext_small()
            self.backbone2 = convnext.convnext_small()
        else:
            self.backbone1 = convnext.convnext_base()
            self.backbone2 = convnext.convnext_base()
        
        self.decode_head = UPerHead_2B(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

    def forward(self, t1, t2):
        fs1 = self.backbone1(t1)[:3]
        fs2 = self.backbone2(t2)[:3]
        fs_diff = []
        for f1, f2 in zip(fs1, fs2):
            fs_diff.append(self.drop(paddle.concat([f1, f2], axis=1)))
        y = self.decode_head(fs_diff)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]

@manager.MODELS.add_component
class CX_Uper_2B_cat(nn.Layer):
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
                    ):
        super(CX_Uper_2B_cat, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone1 = convnext.convnext_tiny()
            self.backbone2 = convnext.convnext_tiny()
        elif backb == 'convnext_small':
            self.backbone1 = convnext.convnext_small()
            self.backbone2 = convnext.convnext_small()
        else:
            self.backbone1 = convnext.convnext_base()
            self.backbone2 = convnext.convnext_base()
        
        self.decode_head = UPerHead_2B(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

    def forward(self, t1, t2):
        fs1 = self.backbone1(t1)[:3]
        fs2 = self.backbone2(t2)[:3]
        fs_diff = []
        for f1, f2 in zip(fs1, fs2):
            fs_diff.append(self.drop(paddle.concat([f1, f2], axis=1)))
        y = self.decode_head(fs_diff)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)
        return [out]

@manager.MODELS.add_component
class CX_Uper_2B_plus(nn.Layer):
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
                    ):
        super(CX_Uper_2B_plus, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone1 = convnext.convnext_tiny()
            self.backbone2 = convnext.convnext_tiny()
        elif backb == 'convnext_small':
            self.backbone1 = convnext.convnext_small()
            self.backbone2 = convnext.convnext_small()
        else:
            self.backbone1 = convnext.convnext_base()
            self.backbone2 = convnext.convnext_base()
        
        self.decode_head1 = UPerHead(self.backbone1.dims[:3], num_classes=num_classes)
        self.decode_head2 = UPerHead(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

    def forward(self, t1, t2):
        fs1 = self.backbone1(t1)[:3]
        fs2 = self.backbone2(t2)[:3]
        y1 = self.decode_head1(fs1)
        y2 = self.decode_head2(fs1)
        y = y1+y2
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)
        return [out]


@manager.MODELS.add_component
class CX_Uper_2B_ca1(nn.Layer):
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
                    ):
        super(CX_Uper_2B_ca1, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone1 = convnext.convnext_tiny()
            self.backbone2 = convnext.convnext_tiny()
        elif backb == 'convnext_small':
            self.backbone1 = convnext.convnext_small()
            self.backbone2 = convnext.convnext_small()
        else:
            self.backbone1 = convnext.convnext_base()
            self.backbone2 = convnext.convnext_base()
        
        self.decode_head = UPerHead(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

        self.cas = []
        for dim in self.backbone1.dims[:3]:
            ca = nn.MultiHeadAttention(dim, 4)
            self.cas.append(ca)


    def forward(self, t1, t2):
        fs1 = self.backbone1(t1)[:3]
        fs2 = self.backbone2(t2)[:3]
        outs = []
        for ca, f1, f2 in zip(self.cas, fs1, fs2):
            shp = f1.shape
            f1 = f1.reshape((shp[0], shp[1], -1)).transpose((0, 2, 1))
            f2 = f2.reshape((shp[0], shp[1], -1)).transpose((0, 2, 1))
            f1 = ca(f1, f2, f1)
            f1 = f1.transpose((0, 2, 1)).reshape((shp[0], shp[1], shp[2], shp[3]))
            outs.append(f1)
        y = self.decode_head(outs)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)
        return [out]

class CX_Uper_2B_ca2(nn.Layer):
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
                    ):
        super(CX_Uper_2B_ca2, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone1 = convnext.convnext_tiny()
            self.backbone2 = convnext.convnext_tiny()
        elif backb == 'convnext_small':
            self.backbone1 = convnext.convnext_small()
            self.backbone2 = convnext.convnext_small()
        else:
            self.backbone1 = convnext.convnext_base()
            self.backbone2 = convnext.convnext_base()
        
        self.decode_head = UPerHead(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

        self.cas = []
        for dim in self.backbone1.dims[:3]:
            ca = nn.MultiHeadAttention(dim, 4)
            self.cas.append(ca)


    def forward(self, t1, t2):
        fs1 = self.backbone1(t1)[:3]
        fs2 = self.backbone2(t2)[:3]
        outs = []
        for ca, f1, f2 in zip(self.cas, fs1, fs2):
            shp = f1.shape
            f1 = f1.reshape((shp[0], shp[1], -1)).transpose((0, 2, 1))
            f2 = f2.reshape((shp[0], shp[1], -1)).transpose((0, 2, 1))
            f2 = ca(f2, f1, f2)
            f2 = f2.transpose((0, 2, 1)).reshape((shp[0], shp[1], shp[2], shp[3]))
            outs.append(f2)
        y = self.decode_head(outs)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)
        return [out]

@manager.MODELS.add_component
class CX_Uper_2B_ca_cat(nn.Layer):
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
                    ):
        super(CX_Uper_2B_ca_cat, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone1 = convnext.convnext_tiny()
            self.backbone2 = convnext.convnext_tiny()
        elif backb == 'convnext_small':
            self.backbone1 = convnext.convnext_small()
            self.backbone2 = convnext.convnext_small()
        else:
            self.backbone1 = convnext.convnext_base()
            self.backbone2 = convnext.convnext_base()
        
        self.decode_head = UPerHead_2B(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

        self.cas = []
        for dim in self.backbone1.dims[:3]:
            ca = nn.MultiHeadAttention(dim, 4)
            self.cas.append(ca)


    def forward(self, t1, t2):
        fs1 = self.backbone1(t1)[:3]
        fs2 = self.backbone2(t2)[:3]
        outs = []
        for ca, f1, f2 in zip(self.cas, fs1, fs2):
            shp = f1.shape
            f1 = f1.reshape((shp[0], shp[1], -1)).transpose((0, 2, 1))
            f2 = f2.reshape((shp[0], shp[1], -1)).transpose((0, 2, 1))
            ff2 = ca(f2, f1, f2)
            ff2 = ff2.transpose((0, 2, 1)).reshape((shp[0], shp[1], shp[2], shp[3]))
            ff1 = ca(f1, f2, f1)
            ff1 = ff1.transpose((0, 2, 1)).reshape((shp[0], shp[1], shp[2], shp[3]))
            outs.append(self.drop(paddle.concat([ff1, ff2], axis=1)))
        y = self.decode_head(outs)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)
        return [out]

@manager.MODELS.add_component
class CX_Uper_2B1(nn.Layer):
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
                    ):
        super(CX_Uper_2B1, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone1 = convnext.convnext_tiny()
        elif backb == 'convnext_small':
            self.backbone1 = convnext.convnext_small()
        else:
            self.backbone1 = convnext.convnext_base()
        
        self.decode_head = UPerHead(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

    def forward(self, t1, t2):
        fs1 = self.backbone1(t1)[:3]
        y = self.decode_head(fs1)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)
        return [out]

@manager.MODELS.add_component
class CX_Uper_3B_cat(nn.Layer):
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
                    ):
        super(CX_Uper_3B_cat, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone1 = convnext.convnext_tiny()
            self.backbone2 = convnext.convnext_tiny()
            self.backbone3 = convnext.convnext_tiny(in_chans=242)
        elif backb == 'convnext_small':
            self.backbone1 = convnext.convnext_small()
            self.backbone2 = convnext.convnext_small()
            self.backbone3 = convnext.convnext_small(in_chans=242)
        else:
            self.backbone1 = convnext.convnext_base()
            self.backbone2 = convnext.convnext_base()
            self.backbone3 = convnext.convnext_base(in_chans=242)
        
        self.decode_head = UPerHead_3B(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)

    def forward(self, t1, t2):
        t3 = t2
        t2 = t1[:, 3:, ...]
        t1 = t1[:, :3, ...]
        fs1 = self.backbone1(t1)
        fs2 = self.backbone2(t2)
        fs3 = self.backbone3(t3)
        fs_diff = []
        for f1, f2, f3 in zip(fs1, fs2, fs3):
            fs_diff.append(self.drop(paddle.concat([f1, f2, f3], axis=1)))
        y = self.decode_head(fs_diff)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]

@manager.MODELS.add_component
class CX_Uper_3B_plus(nn.Layer):
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
                    ckpt_path=None
                    ):
        super(CX_Uper_3B_plus, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone1 = convnext.convnext_tiny()
            self.backbone2 = convnext.convnext_tiny()
            self.backbone3 = convnext.convnext_tiny(in_chans=116)
        elif backb == 'convnext_small':
            self.backbone1 = convnext.convnext_small()
            self.backbone2 = convnext.convnext_small()
            self.backbone3 = convnext.convnext_small(in_chans=116)
        else:
            self.backbone1 = convnext.convnext_base()
            self.backbone2 = convnext.convnext_base()
            self.backbone3 = convnext.convnext_base(in_chans=116)
        
        self.decode_head2b = UPerHead_2B(self.backbone1.dims[:3], num_classes=num_classes)
        self.decode_head3b = UPerHead(self.backbone1.dims[:3], num_classes=num_classes)
        self.drop = nn.Dropout2D(dropout_rate)
        self.ckpt_path = ckpt_path
        self.load_ckpt()
        
    def load_ckpt(self):
        if self.ckpt_path is not None:
            para_state_dict = paddle.load(self.ckpt_path)
            self.set_state_dict(para_state_dict)
            print('load form:', self.ckpt_path)

    def forward(self, t1, t2):
        t3 = t2
        t2 = t1[:, 3:, ...]
        t1 = t1[:, :3, ...]
        fs1 = self.backbone1(t1)
        fs2 = self.backbone2(t2)
        fs3 = self.backbone3(t3)
        fs_diff = []
        for f1, f2 in zip(fs1, fs2):
            fs_diff.append(self.drop(paddle.concat([f1, f2], axis=1)))
        y2 = self.decode_head2b(fs_diff)
        y3 = self.decode_head3b(fs3)
        y = y2+y3
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]


@manager.MODELS.add_component
class CX_Uper(nn.Layer):
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
                    ):
        super(CX_Uper, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone = convnext.convnext_tiny()
        elif backb == 'convnext_small':
            self.backbone = convnext.convnext_small()
        else:
            self.backbone = convnext.convnext_base()
        
        self.decode_head = UPerHead(self.backbone.dims, num_classes=2)
        self.drop = nn.Dropout2D(dropout_rate)

    def forward(self, t1, t2):
        fs1 = self.backbone(t1)
        fs2 = self.backbone(t2)
        fs_diff = []
        for f1, f2 in zip(fs1, fs2):
            fs_diff.append(self.drop(paddle.abs(f1 - f2)))
        y = self.decode_head(fs_diff)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)

        return [out]
@manager.MODELS.add_component
class CX_Uper2(nn.Layer):
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
                    ):
        super(CX_Uper2, self).__init__()
        if backb == 'convnext_tiny':
            self.backbone = convnext.convnext_tiny()
        else:
            self.backbone = convnext.convnext_small()
        
        self.decode_head = UPerHead2(self.backbone.dims, num_classes=2)
        self.drop = nn.Dropout2D(dropout_rate)

    def forward(self, t1, t2):
        fs1 = self.backbone(t1)
        fs2 = self.backbone(t2)
        fs_diff = []
        for f1, f2 in zip(fs1, fs2):
            fs_diff.append(self.drop(paddle.abs(f1 - f2)))
        y, y_aug = self.decode_head(fs_diff)
        out = F.interpolate(y, size=paddle.shape(t1)[2:], mode='bilinear', align_corners=True)
        return [out, y_aug]

class UPerHead2(nn.Layer):
    """Unified Perceptual Parsing for Scene Understanding
    https://arxiv.org/abs/1807.10221
    scales: Pooling scales used in PPM module applied on the last feature
    """
    def __init__(self, in_channels, channel=128, num_classes: int = 19, scales=(1, 2, 3, 6)):
        super().__init__()
        # PPM Module
        self.ppm = PPM(in_channels[-1], channel, scales)

        # FPN Module
        self.fpn_in = nn.LayerList()
        self.fpn_out = nn.LayerList()

        for in_ch in in_channels[:-1]: # skip the top layer
            self.fpn_in.append(ConvModule(in_ch, channel, 1))
            self.fpn_out.append(ConvModule(channel, channel, 3, 1, 1))

        self.bottleneck = ConvModule(len(in_channels)*channel, channel, 3, 1, 1)
        self.dropout = nn.Dropout2D(0.1)
        self.conv_seg = nn.Conv2D(channel, num_classes, 1)
        self.conv_seg_aug = nn.Conv2DTranspose(128, 2, kernel_size=2, stride=2)


    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        f = self.ppm(features[-1])
        fpn_features = [f]

        for i in reversed(range(len(features)-1)):
            feature = self.fpn_in[i](features[i])
            f = feature + F.interpolate(f, size=feature.shape[-2:], mode='bilinear', align_corners=False)
            fpn_features.append(self.fpn_out[i](f))

        fpn_features.reverse()
        for i in range(1, len(features)):
            fpn_features[i] = F.interpolate(fpn_features[i], size=fpn_features[0].shape[-2:], mode='bilinear', align_corners=False)
 
        output_feature = self.bottleneck(paddle.concat(fpn_features, axis=1))
        output_aug = self.conv_seg_aug(output_feature)
        output = self.conv_seg(self.dropout(output_feature))

        return output, output_aug

class UPerHead_2B(nn.Layer):
    """Unified Perceptual Parsing for Scene Understanding
    https://arxiv.org/abs/1807.10221
    scales: Pooling scales used in PPM module applied on the last feature
    """
    def __init__(self, in_channels, channel=128, num_classes: int = 19, scales=(1, 2, 3, 6)):
        super().__init__()
        in_channels = [ch*2 for ch in in_channels]
        # PPM Module
        self.ppm = PPM(in_channels[-1], channel, scales)

        # FPN Module
        self.fpn_in = nn.LayerList()
        self.fpn_out = nn.LayerList()

        for in_ch in in_channels[:-1]: # skip the top layer
            self.fpn_in.append(ConvModule(in_ch, channel, 1))
            self.fpn_out.append(ConvModule(channel, channel, 3, 1, 1))

        self.bottleneck = ConvModule(len(in_channels)*channel, channel, 3, 1, 1)
        self.dropout = nn.Dropout2D(0.1)
        self.conv_seg = nn.Conv2D(channel, num_classes, 1)


    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        f = self.ppm(features[-1])
        fpn_features = [f]

        for i in reversed(range(len(features)-1)):
            feature = self.fpn_in[i](features[i])
            f = feature + F.interpolate(f, size=feature.shape[-2:], mode='bilinear', align_corners=False)
            fpn_features.append(self.fpn_out[i](f))

        fpn_features.reverse()
        for i in range(1, len(features)):
            fpn_features[i] = F.interpolate(fpn_features[i], size=fpn_features[0].shape[-2:], mode='bilinear', align_corners=False)
 
        output = self.bottleneck(paddle.concat(fpn_features, axis=1))
        output = self.conv_seg(self.dropout(output))
        return output

class UPerHead_3B(nn.Layer):
    """Unified Perceptual Parsing for Scene Understanding
    https://arxiv.org/abs/1807.10221
    scales: Pooling scales used in PPM module applied on the last feature
    """
    def __init__(self, in_channels, channel=128, num_classes: int = 19, scales=(1, 2, 3, 6)):
        super().__init__()
        in_channels = [ch*3 for ch in in_channels]
        # PPM Module
        self.ppm = PPM(in_channels[-1], channel, scales)

        # FPN Module
        self.fpn_in = nn.LayerList()
        self.fpn_out = nn.LayerList()

        for in_ch in in_channels[:-1]: # skip the top layer
            self.fpn_in.append(ConvModule(in_ch, channel, 1))
            self.fpn_out.append(ConvModule(channel, channel, 3, 1, 1))

        self.bottleneck = ConvModule(len(in_channels)*channel, channel, 3, 1, 1)
        self.dropout = nn.Dropout2D(0.1)
        self.conv_seg = nn.Conv2D(channel, num_classes, 1)


    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        f = self.ppm(features[-1])
        fpn_features = [f]

        for i in reversed(range(len(features)-1)):
            feature = self.fpn_in[i](features[i])
            f = feature + F.interpolate(f, size=feature.shape[-2:], mode='bilinear', align_corners=False)
            fpn_features.append(self.fpn_out[i](f))

        fpn_features.reverse()
        for i in range(1, len(features)):
            fpn_features[i] = F.interpolate(fpn_features[i], size=fpn_features[0].shape[-2:], mode='bilinear', align_corners=False)
 
        output = self.bottleneck(paddle.concat(fpn_features, axis=1))
        output = self.conv_seg(self.dropout(output))
        return output


class UPerHead(nn.Layer):
    """Unified Perceptual Parsing for Scene Understanding
    https://arxiv.org/abs/1807.10221
    scales: Pooling scales used in PPM module applied on the last feature
    """
    def __init__(self, in_channels, channel=128, num_classes: int = 19, scales=(1, 2, 3, 6)):
        super().__init__()
        # PPM Module
        self.ppm = PPM(in_channels[-1], channel, scales)

        # FPN Module
        self.fpn_in = nn.LayerList()
        self.fpn_out = nn.LayerList()

        for in_ch in in_channels[:-1]: # skip the top layer
            self.fpn_in.append(ConvModule(in_ch, channel, 1))
            self.fpn_out.append(ConvModule(channel, channel, 3, 1, 1))

        self.bottleneck = ConvModule(len(in_channels)*channel, channel, 3, 1, 1)
        self.dropout = nn.Dropout2D(0.1)
        self.conv_seg = nn.Conv2D(channel, num_classes, 1)


    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        f = self.ppm(features[-1])
        fpn_features = [f]

        for i in reversed(range(len(features)-1)):
            feature = self.fpn_in[i](features[i])
            f = feature + F.interpolate(f, size=feature.shape[-2:], mode='bilinear', align_corners=False)
            fpn_features.append(self.fpn_out[i](f))

        fpn_features.reverse()
        for i in range(1, len(features)):
            fpn_features[i] = F.interpolate(fpn_features[i], size=fpn_features[0].shape[-2:], mode='bilinear', align_corners=False)
 
        output = self.bottleneck(paddle.concat(fpn_features, axis=1))
        output = self.conv_seg(self.dropout(output))
        return output

class PPM(nn.Layer):
    """Pyramid Pooling Module in PSPNet
    """
    def __init__(self, c1, c2=128, scales=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.LayerList([
            nn.Sequential(
                nn.AdaptiveAvgPool2D(scale),
                ConvModule(c1, c2, 1)
            )
        for scale in scales])

        self.bottleneck = ConvModule(c1 + c2 * len(scales), c2, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        outs = []
        for stage in self.stages:
            outs.append(F.interpolate(stage(x), size=x.shape[-2:], mode='bilinear', align_corners=True))

        outs = [x] + outs[::-1]
        out = self.bottleneck(paddle.concat(outs, axis=1))
        return out

class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2D(c1, c2, k, s, p, d, g, bias_attr=False),
            nn.BatchNorm2D(c2),
            nn.ReLU(True)
        )