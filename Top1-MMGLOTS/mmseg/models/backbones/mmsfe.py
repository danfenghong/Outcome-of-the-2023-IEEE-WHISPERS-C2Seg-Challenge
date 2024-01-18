import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from ..builder import BACKBONES
from .beit import BEiT
from .cnns import HSICNN, SARCNN


@BACKBONES.register_module()
class MMSFE(BaseModule):
    def __init__(self,
                img_size=224,
                patch_size=16,
                in_channels=3,
                embed_dims=768,
                num_layers=12,
                num_heads=12,
                mlp_ratio=4,
                out_indices=-1,
                qv_bias=True,
                attn_drop_rate=0.,
                drop_path_rate=0.,
                norm_cfg=dict(type='LN'),
                act_cfg=dict(type='GELU'),
                patch_norm=False,
                final_norm=False,
                num_fcs=2,
                norm_eval=False,
                pretrained=None,
                init_values=0.1,
                init_cfg=None,
                hsi_channels=242):
        super().__init__()

        self.transformer_extractor = BEiT(img_size,
                                          patch_size,
                                          in_channels,
                                          embed_dims,
                                          num_layers,
                                          num_heads,
                                          mlp_ratio,
                                          out_indices,
                                          qv_bias,
                                          attn_drop_rate,
                                          drop_path_rate,
                                          norm_cfg,
                                          act_cfg,
                                          patch_norm,
                                          final_norm,
                                          num_fcs,
                                          norm_eval,
                                          pretrained,
                                          init_values,
                                          init_cfg)
        self.hsi_cnn = HSICNN(hsi_channels=hsi_channels)
        self.sar_cnn = SARCNN()

    def forward(self, x):
        msi, hsi, sar = x
        msi_feat = self.transformer_extractor(msi) # 768x40x40
        hsi_feat = self.hsi_cnn(hsi) # 64x160x160; 128x80x80; 256x40x40; 512x20x20
        sar_feat = self.sar_cnn(sar)

        return (msi_feat, hsi_feat, sar_feat)