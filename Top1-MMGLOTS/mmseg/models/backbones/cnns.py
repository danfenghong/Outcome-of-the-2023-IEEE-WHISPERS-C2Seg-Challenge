import torch.nn as nn
from .resnet import ResNetV1c

class HSICNN(nn.Module):
    def __init__(self, hsi_channels=242):
        super().__init__()
        self.hsi_cnn = ResNetV1c(
            depth=34,
            in_channels=hsi_channels)
            # out_indices=(0, 1, 2, 3),
        self.out1_smooth = nn.Conv2d(in_channels=64, out_channels=768, kernel_size=1)
        self.out2_smooth = nn.Conv2d(in_channels=128, out_channels=768, kernel_size=1)
        self.out3_smooth = nn.Conv2d(in_channels=256, out_channels=768, kernel_size=1)
        self.out4_smooth = nn.Conv2d(in_channels=512, out_channels=768, kernel_size=1)

    def forward(self, x):
        o = []
        x = self.hsi_cnn(x)
        o.append(self.out1_smooth(x[0]))
        o.append(self.out2_smooth(x[1]))
        o.append(self.out3_smooth(x[2]))
        o.append(self.out4_smooth(x[3]))
        return o
    
class SARCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.sar_cnn = ResNetV1c(
            depth=34,
            in_channels=2)
        self.out1_smooth = nn.Conv2d(in_channels=64, out_channels=768, kernel_size=1)
        self.out2_smooth = nn.Conv2d(in_channels=128, out_channels=768, kernel_size=1)
        self.out3_smooth = nn.Conv2d(in_channels=256, out_channels=768, kernel_size=1)
        self.out4_smooth = nn.Conv2d(in_channels=512, out_channels=768, kernel_size=1)

    def forward(self, x):
        o = []
        x = self.sar_cnn(x)
        o.append(self.out1_smooth(x[0]))
        o.append(self.out2_smooth(x[1]))
        o.append(self.out3_smooth(x[2]))
        o.append(self.out4_smooth(x[3]))
        return o 