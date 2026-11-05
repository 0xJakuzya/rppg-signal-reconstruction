import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

class PhysNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvBlock1 = Conv(3, 16, (1, 5, 5), (0, 2, 2))
        self.ConvBlock2 = Conv(16, 32)
        self.ConvBlock3 = Conv(32, 64)
        self.ConvBlock4 = Conv(64, 64)
        self.ConvBlock5 = Conv(64, 64)
        self.ConvBlock6 = Conv(64, 64)
        self.ConvBlock7 = Conv(64, 64)
        self.ConvBlock8 = Conv(64, 64)
        self.ConvBlock9 = Conv(64, 64)
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(64, 64, (4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, (4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.ConvBlock10 = nn.Conv3d(64, 1, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        length = x.shape[2]
        
        x = self.MaxpoolSpa(self.ConvBlock1(x))
        x = self.ConvBlock2(x)
        x = self.MaxpoolSpaTem(self.ConvBlock3(x))

        x = self.ConvBlock4(x)
        x = self.MaxpoolSpaTem(self.ConvBlock5(x))

        x = self.ConvBlock6(x)
        x = self.MaxpoolSpa(self.ConvBlock7(x))

        x = self.ConvBlock8(x)
        x = self.ConvBlock9(x)

        x = self.upsample(x)
        x = self.upsample2(x)

        x = F.adaptive_avg_pool3d(x, (length, 1, 1))
        x = self.ConvBlock10(x)

        return x.view(-1, length)
