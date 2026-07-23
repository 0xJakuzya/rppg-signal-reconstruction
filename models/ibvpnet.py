""" iBVPNet
original source:
iBVP Dataset: RGB-Thermal rPPG Dataset with High Resolution Signal Quality Labels
Electronics 2024, 13(7), 1334
By Jitesh Joshi and Youngjun Cho
"""
import torch
import torch.nn as nn

from src import config


class ConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super().__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_3d(x)


class DeConvBlock3D(nn.Module):
    """Upsamples time and space in two separate steps."""
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super().__init__()
        k_t, k_s1, k_s2 = kernel_size
        s_t, s_s1, s_s2 = stride
        self.deconv_block_3d = nn.Sequential(
            nn.ConvTranspose3d(in_channel, in_channel, (k_t, 1, 1), (s_t, 1, 1), padding),
            nn.Tanh(),
            nn.InstanceNorm3d(in_channel),

            nn.Conv3d(in_channel, out_channel, (1, k_s1, k_s2), (1, s_s1, s_s2), padding),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.deconv_block_3d(x)


class encoder_block(nn.Module):
    """Spatial features first, then long temporal kernels (11 and 7 frames)."""
    def __init__(self, in_channel, nf):
        super().__init__()
        self.spatio_temporal_encoder = nn.Sequential(
            ConvBlock3D(in_channel, nf[0], [1, 3, 3], [1, 1, 1], [0, 1, 1]),
            ConvBlock3D(nf[0], nf[1], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            ConvBlock3D(nf[1], nf[2], [1, 3, 3], [1, 1, 1], [0, 1, 1]),
            ConvBlock3D(nf[2], nf[3], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            ConvBlock3D(nf[3], nf[4], [1, 3, 3], [1, 1, 1], [0, 1, 1]),
            ConvBlock3D(nf[4], nf[4], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
        )

        self.temporal_encoder = nn.Sequential(
            ConvBlock3D(nf[4], nf[4], [11, 1, 1], [1, 1, 1], [5, 0, 0]),
            ConvBlock3D(nf[4], nf[4], [11, 3, 3], [1, 1, 1], [5, 1, 1]),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
            ConvBlock3D(nf[4], nf[4], [11, 1, 1], [1, 1, 1], [5, 0, 0]),
            ConvBlock3D(nf[4], nf[4], [11, 3, 3], [1, 1, 1], [5, 1, 1]),
            nn.MaxPool3d((2, 2, 2), stride=(2, 1, 1)),
            ConvBlock3D(nf[4], nf[4], [7, 1, 1], [1, 1, 1], [3, 0, 0]),
            ConvBlock3D(nf[4], nf[4], [7, 3, 3], [1, 1, 1], [3, 1, 1]),
        )

    def forward(self, x):
        return self.temporal_encoder(self.spatio_temporal_encoder(x))


class decoder_block(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.decoder_block = nn.Sequential(
            DeConvBlock3D(nf[4], nf[3], [7, 3, 3], [2, 2, 2], [2, 1, 1]),
            DeConvBlock3D(nf[3], nf[2], [7, 3, 3], [2, 2, 2], [2, 1, 1]),
        )

    def forward(self, x):
        return self.decoder_block(x)


class iBVPNet(nn.Module):
    """3D-CNN mapping raw full-face rPPG windows to a 1D BVP signal.

    Takes [B, T, 3, H, W] and returns [B, T], matching PhysNet so the existing
    training loop and ShiftLoss apply unchanged. Spatial size is free: the
    adaptive pool collapses it whatever it is.
    """
    def __init__(self, in_channels=None, nf=None):
        super().__init__()
        in_channels = config.ibvp_in_channels if in_channels is None else in_channels
        nf = config.ibvp_channels if nf is None else nf

        self.in_channels = in_channels
        self.norm = nn.InstanceNorm3d(in_channels)
        self.encoder = encoder_block(in_channels, nf)
        self.decoder = decoder_block(nf)
        self.head = nn.Conv3d(nf[2], 1, [1, 1, 1], stride=1, padding=0)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()   # [B, C, T, H, W]
        length = x.shape[2]

        # Diff as first layer removes stationary components; it costs one frame,
        # so pad the front to keep the caller's window length.
        x = torch.diff(x, dim=2)
        x = self.norm(x[:, :self.in_channels, :, :, :])

        x = self.decoder(self.encoder(x))
        # The encoder halves time twice and the decoder doubles it twice, but
        # odd lengths do not round-trip exactly -- pool back to what diff left.
        x = nn.functional.adaptive_max_pool3d(x, (length - 1, 1, 1))
        x = self.head(x)

        rPPG = x.view(-1, length - 1)
        rPPG = torch.cat([rPPG[:, :1], rPPG], dim=1)
        return rPPG
