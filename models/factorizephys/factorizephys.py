""" FactorizePhys
original source:
FactorizePhys: Matrix Factorization for Multidimensional Attention in Remote
Physiological Sensing
Advances in Neural Information Processing Systems (NeurIPS) 2024,
By Jitesh Joshi, Sos S. Agaian, and Youngjun Cho
"""
import torch
import torch.nn as nn
from src import config
from .fsam import FeaturesFactorizationModule

class ConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super().__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding=padding, bias=False),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_3d(x)


class rPPG_FeatureExtractor(nn.Module):

    def __init__(self, in_channel, nf, dropout_rate):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ConvBlock3D(in_channel, nf[0], [3, 3, 3], [1, 1, 1], [1, 1, 1]),  # T, 72, 72
            ConvBlock3D(nf[0], nf[1], [3, 3, 3], [1, 2, 2], [1, 0, 0]),       # T, 35, 35
            ConvBlock3D(nf[1], nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]),       # T, 33, 33
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf[1], nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]),       # T, 31, 31
            ConvBlock3D(nf[1], nf[2], [3, 3, 3], [1, 2, 2], [1, 0, 0]),       # T, 15, 15
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]),       # T, 13, 13
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x):
        return self.feature_extractor(x)


class BVP_Head(nn.Module):

    def __init__(self, nf, dropout_rate, use_fsam, md_residual):
        super().__init__()
        self.use_fsam = use_fsam
        self.md_res = md_residual

        self.conv_block = nn.Sequential(
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]),  # T, 11, 11
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]),  # T, 9, 9
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]),  # T, 7, 7
            nn.Dropout3d(p=dropout_rate),
        )

        if self.use_fsam:
            self.fsam = FeaturesFactorizationModule(nf[2])
            self.fsam_norm = nn.InstanceNorm3d(nf[2])
            self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.final_layer = nn.Sequential(
            ConvBlock3D(nf[2], nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]),   # T, 5, 5
            ConvBlock3D(nf[1], nf[0], [3, 3, 3], [1, 1, 1], [1, 0, 0]),   # T, 3, 3
            nn.Conv3d(nf[0], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),  # T, 1, 1
        )

    def forward(self, voxel_embeddings, length):
        voxel_embeddings = self.conv_block(voxel_embeddings)

        if self.use_fsam:
            att_mask, appx_error = self.fsam(voxel_embeddings - voxel_embeddings.min())
            x = torch.mul(voxel_embeddings - voxel_embeddings.min() + self.bias1,
                          att_mask - att_mask.min() + self.bias1)
            factorized_embeddings = self.fsam_norm(x)
            if self.md_res:
                factorized_embeddings = voxel_embeddings + factorized_embeddings
            x = self.final_layer(factorized_embeddings)
        else:
            appx_error = None
            x = self.final_layer(voxel_embeddings)

        rPPG = x.view(-1, length)
        return rPPG, appx_error


class FactorizePhys(nn.Module):
    """3D-CNN mapping raw full-face rPPG windows to a 1D BVP signal."""
    def __init__(self, in_channels=None, nf=None, dropout=None,
                 use_fsam=None, md_residual=None):
        super().__init__()
        in_channels = config.fp_in_channels if in_channels is None else in_channels
        nf = config.fp_channels if nf is None else nf
        dropout = config.fp_dropout if dropout is None else dropout
        use_fsam = config.fp_use_fsam if use_fsam is None else use_fsam
        md_residual = config.fp_md_residual if md_residual is None else md_residual

        self.in_channels = in_channels
        self.norm = nn.InstanceNorm3d(in_channels)
        self.rppg_feature_extractor = rPPG_FeatureExtractor(in_channels, nf, dropout)
        self.rppg_head = BVP_Head(nf, dropout, use_fsam, md_residual)
        self.last_appx_error = None

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        length = x.shape[2]
        x = torch.diff(x, dim=2)
        x = self.norm(x[:, :self.in_channels, :, :, :])

        voxel_embeddings = self.rppg_feature_extractor(x)
        rPPG, appx_error = self.rppg_head(voxel_embeddings, length - 1)
        self.last_appx_error = appx_error

        rPPG = torch.cat([rPPG[:, :1], rPPG], dim=1)
        return rPPG
