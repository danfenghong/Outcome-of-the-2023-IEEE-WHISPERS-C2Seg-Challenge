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

from .bit import *
from .dsfin import *



@manager.MODELS.add_component
class BIDSF(nn.Layer):

    def __init__(self,
                 in_channels,
                 num_classes,
                #  backbone,
                 backb,
                 n_stages=4,
                 use_tokenizer=True,
                 token_len=4,
                 pool_mode='max',
                 pool_size=2,
                 enc_with_pos=True,
                 enc_depth=1,
                 enc_head_dim=64,
                 dec_depth=8,
                 dec_head_dim=8,
                 EBD_DIM=32,
                 use_dropout=False,
                 features_bool=True,
                 **backbone_kwargs):
        super(BIDSF, self).__init__()

        # TODO: reduce hard-coded parameters
        DIM = 32
        MLP_DIM = 2 * DIM
        EBD_DIM = DIM

        if backb == 'vgg16':
            self.encoder1 = self.encoder2 = VGG16FeaturePicker()
        else:
            self.encoder1 = self.encoder2 = ResNetFeaturePicker(arch=backb, pretrained=True, features_bool=features_bool)

        self.use_tokenizer = use_tokenizer
        if not use_tokenizer:
            # If a tokenzier is not to be used，then downsample the feature maps.
            self.pool_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = pool_size * pool_size
        else:
            self.conv_att = Conv1x1(32, token_len, bias=False)
            self.token_len = token_len

        self.enc_with_pos = enc_with_pos
        if enc_with_pos:
            self.enc_pos_embedding = self.create_parameter(
                shape=(1, self.token_len * 2, EBD_DIM),
                default_initializer=Normal())

        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.enc_head_dim = enc_head_dim
        self.dec_head_dim = dec_head_dim

        self.encoder = TransformerEncoder(
            dim=DIM,
            depth=enc_depth,
            n_heads=8,
            head_dim=enc_head_dim,
            mlp_dim=MLP_DIM,
            dropout_rate=0.)
        self.decoder = TransformerDecoder(
            dim=DIM,
            depth=dec_depth,
            n_heads=8,
            head_dim=dec_head_dim,
            mlp_dim=MLP_DIM,
            dropout_rate=0.,
            apply_softmax=True)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.conv_out = nn.Sequential(
            Conv3x3(
                EBD_DIM, EBD_DIM, norm=True, act=True),
            Conv3x3(EBD_DIM, num_classes))

        self.down_conv = nn.Conv2D(64, 128, kernel_size=2, stride=2)
        self.up_conv = nn.Conv2DTranspose(256, 128, kernel_size=2, stride=2)

        self.sa3 = SpatialAttention()
        self.ca3 = ChannelAttention(in_ch=384)
        self.o3_conv1 = conv2d_bn(384, 128, use_dropout)
        self.o3_conv2 = conv2d_bn(128, 128, use_dropout)
        self.o3_conv3 = conv2d_bn(128, 32, use_dropout)
        self.bn_sa3 = make_norm(32)
        self.o3_conv4 = Conv1x1(32, num_classes)

        self.sa5 = SpatialAttention()
        self.ca5 = ChannelAttention(in_ch=192)
        self.o5_conv1 = conv2d_bn(192, 64, use_dropout)
        self.o5_conv2 = conv2d_bn(64, 32, use_dropout)
        self.o5_conv3 = conv2d_bn(32, 16, use_dropout)
        self.bn_sa5 = make_norm(16)

    def _get_semantic_tokens(self, x):
        b, c = paddle.shape(x)[:2]
        att_map = self.conv_att(x)
        att_map = att_map.reshape((b, self.token_len, 1, -1))
        att_map = F.softmax(att_map, axis=-1)
        x = x.reshape((b, 1, c, -1))
        tokens = (x * att_map).sum(-1)
        return tokens

    def _get_reshaped_tokens(self, x):
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, (self.pool_size, self.pool_size))
        elif self.pool_mode == 'avg':
            x = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        else:
            x = x
        tokens = x.transpose((0, 2, 3, 1)).flatten(1, 2)
        return tokens

    def encode(self, x):
        if self.enc_with_pos:
            x += self.enc_pos_embedding
        x = self.encoder(x)
        return x

    def decode(self, x, m):
        b, c, h, w = paddle.shape(x)
        x = x.transpose((0, 2, 3, 1)).flatten(1, 2)
        x = self.decoder(x, m)
        x = x.transpose((0, 2, 1)).reshape((b, c, h, w))
        return x

    def forward(self, t1, t2):
        with paddle.no_grad():
            self.encoder1.eval(), self.encoder2.eval()
            t1_feats = self.encoder1(t1)
            t2_feats = self.encoder2(t2)
        t1_s0, t1_s1, t1_s2, t1_s3, _ = t1_feats
        t2_s0, t2_s1, t2_s2, t2_s3, _ = t2_feats

        # print(t1_f_l3.shape, t1_f_l8.shape, t1_f_l15.shape, t1_f_l22.shape, t1_f_l29)
        t1_s1 = self.down_conv(t1_s1)
        t2_s1 = self.down_conv(t2_s1)
        t1_s3 = self.up_conv(t1_s3)
        t2_s3 = self.up_conv(t2_s3)
        x1 = paddle.concat([t1_s1, t1_s2, t1_s3], axis=1)
        x2 = paddle.concat([t2_s1, t2_s2, t2_s3], axis=1)


        x1 = self.ca3(x1) * x1
        x1 = self.o3_conv1(x1)
        x1 = self.o3_conv2(x1)
        x1 = self.o3_conv3(x1)
        x1 = self.sa3(x1) * x1

        x2 = self.ca3(x2) * x2
        x2 = self.o3_conv1(x2)
        x2 = self.o3_conv2(x2)
        x2 = self.o3_conv3(x2)
        x2 = self.sa3(x2) * x2

        y3 = paddle.abs(x1 - x2)
        pred_aug = self.o3_conv4(y3)
        pred = F.interpolate(
                pred_aug,
                paddle.shape(t1)[2:],
                mode='bilinear',
                align_corners=False)


        # Tokenization
        if self.use_tokenizer:
            token1 = self._get_semantic_tokens(x1)
            token2 = self._get_semantic_tokens(x2)
        else:
            token1 = self._get_reshaped_tokens(x1)
            token2 = self._get_reshaped_tokens(x2)

        # Transformer encoder forward
        token = paddle.concat([token1, token2], axis=1)
        token = self.encode(token)
        token1, token2 = paddle.chunk(token, 2, axis=1)

        # Transformer decoder forward
        y1 = self.decode(x1, token1)
        y2 = self.decode(x2, token2)

        # Feature differencing
        y = paddle.abs(y1 - y2)
        y = self.upsample(y)

        # Classifier forward
        pred = self.conv_out(y)
        pred = F.interpolate(
                pred,
                paddle.shape(t1)[2:],
                mode='bilinear',
                align_corners=False)
        return [pred, pred]

    def init_weight(self):
        # Use the default initialization method.
        pass


