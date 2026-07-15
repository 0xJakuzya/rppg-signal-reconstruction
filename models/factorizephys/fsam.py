""" FSAM: Factorized Self-Attention Module
original source:
FactorizePhys: Matrix Factorization for Multidimensional Attention in Remote
Physiological Sensing, NeurIPS 2024
By Jitesh Joshi, Sos S. Agaian, and Youngjun Cho
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src import config

class NMF3D(nn.Module):
    """Rank-R non-negative matrix factorization over voxel embeddings.
    Iterative multiplicative updates run under no_grad: gradients never flow
    through the factorization, only through the pre/post projections.
    """
    def __init__(self, slices=None, rank=None, steps=None):
        super().__init__()
        self.S = config.fp_md_slices if slices is None else slices
        self.R = config.fp_md_rank if rank is None else rank
        self.steps = config.fp_md_steps if steps is None else steps
        self.inv_t = 1

    def _build_bases(self, batch, S, D, R, device, dtype):
        bases = torch.ones((batch * S, D, R), device=device, dtype=dtype)
        return F.normalize(bases, dim=1)

    @torch.no_grad()
    def _local_step(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)

        numerator = torch.bmm(x, coef)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * numerator / (denominator + 1e-6)
        return bases, coef

    @torch.no_grad()
    def _local_inference(self, x, bases):
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)
        for _ in range(self.steps):
            bases, coef = self._local_step(x, bases, coef)
        return bases, coef

    def _compute_coef(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        return coef * numerator / (denominator + 1e-6)

    def forward(self, x):
        batch, channels, length, height, width = x.shape
        out_dtype = x.dtype
        x = x.float()
        D = length // self.S
        N = channels * height * width
        x = x.view(batch * self.S, D, N)
        bases = self._build_bases(batch, self.S, D, self.R, x.device, x.dtype)
        bases, coef = self._local_inference(x, bases)
        coef = self._compute_coef(x, bases, coef)
        x = torch.bmm(bases, coef.transpose(1, 2))
        return x.view(batch, channels, length, height, width).to(out_dtype)


class ConvBNReLU3D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FeaturesFactorizationModule(nn.Module):

    def __init__(self, in_channel, align_channel=None):
        super().__init__()
        align_channel = config.fp_align_channels if align_channel is None else align_channel

        self.pre_conv_block = nn.Sequential(
            nn.Conv3d(in_channel, align_channel, (1, 1, 1)),
            nn.ReLU(inplace=True),
        )
        self.md_block = NMF3D()
        self.post_conv_block = nn.Sequential(
            ConvBNReLU3D(align_channel, align_channel),
            nn.Conv3d(align_channel, in_channel, 1, bias=False),
        )
        self._init_weight()

    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                fan_out = (module.kernel_size[0] * module.kernel_size[1]
                           * module.kernel_size[2] * module.out_channels)
                module.weight.data.normal_(0, np.sqrt(2.0 / fan_out))
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        x = self.pre_conv_block(x)
        with torch.autocast("cuda", enabled=False):
            attention = self.md_block(x)
        approx_error = torch.dist(x.float(), attention.float())
        attention = self.post_conv_block(attention)
        return attention, approx_error
