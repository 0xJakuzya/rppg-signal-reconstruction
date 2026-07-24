"""RhythmFormer adapted to the local rPPG training interface.

The architecture is based on:
    B. Zou et al., "RhythmFormer: Extracting Patterned rPPG Signals Based on
    Periodic Sparse Attention", Pattern Recognition 164 (2025), 111511.

Upstream implementation:
    https://github.com/zizheng-guo/RhythmFormer

The upstream code is MIT licensed:

Copyright (c) 2023 Zizheng Guo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Local adaptations:
  * no dependency on timm;
  * configurable parameters are read from src.config;
  * temporal padding lets the original (2, 4, 8) hierarchy accept T=300;
  * input/output match the project convention [B,T,3,H,W] -> [B,T].
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src import config


class DropPath(nn.Module):
    """Per-sample stochastic depth, equivalent to timm's DropPath."""

    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep_probability = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_probability)
        return x * mask / keep_probability


class FusionStem(nn.Module):
    """Fuse raw RGB appearance with four local temporal differences."""

    def __init__(self, appearance_weight: float = 0.5, difference_weight: float = 0.5):
        super().__init__()
        self.appearance_path = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.difference_path = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.fused_path = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.refined_difference_path = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.appearance_weight = float(appearance_weight)
        self.difference_weight = float(difference_weight)

    def forward(self, x: Tensor) -> Tensor:
        batch, frames, channels, height, width = x.shape
        if channels != 3:
            raise ValueError(f"RhythmFormer expects RGB input, got {channels} channels")
        if frames < 3:
            raise ValueError("RhythmFormer needs at least three frames")

        two_back = torch.cat((x[:, :1], x[:, :1], x[:, :-2]), dim=1)
        one_back = torch.cat((x[:, :1], x[:, :-1]), dim=1)
        one_ahead = torch.cat((x[:, 1:], x[:, -1:]), dim=1)
        two_ahead = torch.cat((x[:, 2:], x[:, -1:], x[:, -1:]), dim=1)
        differences = torch.cat(
            (one_back - two_back, x - one_back, one_ahead - x, two_ahead - one_ahead),
            dim=2,
        )

        difference_features = self.difference_path(
            differences.reshape(batch * frames, 12, height, width)
        )
        appearance_features = self.appearance_path(
            x.reshape(batch * frames, channels, height, width)
        )
        fused = (
            self.appearance_weight * appearance_features
            + self.difference_weight * difference_features
        )
        fused = self.fused_path(fused)
        difference_features = self.refined_difference_path(difference_features)
        return (
            self.appearance_weight * fused
            + self.difference_weight * difference_features
        )


class TemporalCentralDifference(nn.Module):
    """Temporal central-difference convolution used by RhythmFormer."""

    def __init__(self, channels: int, theta: float = 0.2):
        super().__init__()
        self.conv = nn.Conv3d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.theta = float(theta)

    def forward(self, x: Tensor) -> Tensor:
        normal = self.conv(x)
        if abs(self.theta) < 1e-8:
            return normal
        weight = self.conv.weight
        temporal_difference = (
            weight[:, :, 0].sum(dim=(-1, -2))
            + weight[:, :, 2].sum(dim=(-1, -2))
        )[:, :, None, None, None]
        difference = F.conv3d(x, temporal_difference)
        return normal - self.theta * difference


def _grid_to_sequence(
    x: Tensor, region_size: tuple[int, int, int], num_heads: int
) -> tuple[Tensor, int, int, int]:
    batch, channels, frames, height, width = x.shape
    region_t = frames // region_size[0]
    region_h = height // region_size[1]
    region_w = width // region_size[2]
    x = x.reshape(
        batch,
        num_heads,
        channels // num_heads,
        region_t,
        region_size[0],
        region_h,
        region_size[1],
        region_w,
        region_size[2],
    )
    x = torch.einsum("bmdtohpwq->bmthwopqd", x)
    x = x.flatten(2, 4).flatten(-4, -2)
    return x, region_t, region_h, region_w


def _sequence_to_grid(
    x: Tensor,
    region_t: int,
    region_h: int,
    region_w: int,
    region_size: tuple[int, int, int],
) -> Tensor:
    batch, heads, _, _, head_dim = x.shape
    x = x.reshape(
        batch,
        heads,
        region_t,
        region_h,
        region_w,
        region_size[0],
        region_size[1],
        region_size[2],
        head_dim,
    )
    x = torch.einsum("bmthwopqd->bmdtohpwq", x)
    return x.reshape(
        batch,
        heads * head_dim,
        region_t * region_size[0],
        region_h * region_size[1],
        region_w * region_size[2],
    )


def regional_routing_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
    region_graph: Tensor,
    region_size: tuple[int, int, int],
) -> Tensor:
    batch, heads, query_regions, topk = region_graph.shape
    query, region_t, region_h, region_w = _grid_to_sequence(
        query, region_size, heads
    )
    key, _, _, _ = _grid_to_sequence(key, region_size, heads)
    value, _, _, _ = _grid_to_sequence(value, region_size, heads)

    _, _, key_regions, key_region_volume, head_dim = key.shape
    gather_index = region_graph.reshape(
        batch, heads, query_regions, topk, 1, 1
    ).expand(-1, -1, -1, -1, key_region_volume, head_dim)
    key = torch.gather(
        key.reshape(batch, heads, 1, key_regions, key_region_volume, head_dim).expand(
            -1, -1, query_regions, -1, -1, -1
        ),
        dim=3,
        index=gather_index,
    )
    value = torch.gather(
        value.reshape(
            batch, heads, 1, key_regions, key_region_volume, head_dim
        ).expand(-1, -1, query_regions, -1, -1, -1),
        dim=3,
        index=gather_index,
    )
    attention = (query * scale) @ key.flatten(-3, -2).transpose(-1, -2)
    attention = torch.softmax(attention, dim=-1)
    output = attention @ value.flatten(-3, -2)
    return _sequence_to_grid(
        output, region_t, region_h, region_w, region_size
    )


class PeriodicSparseAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        temporal_patch: int,
        topk: int,
        side_kernel: int,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError("embedding dimension must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = dim ** -0.5
        self.temporal_patch = temporal_patch
        self.topk = topk
        self.local_context = nn.Conv3d(
            dim,
            dim,
            kernel_size=side_kernel,
            padding=side_kernel // 2,
            groups=dim,
        )
        self.query = nn.Sequential(
            TemporalCentralDifference(dim),
            nn.BatchNorm3d(dim),
        )
        self.key = nn.Sequential(
            TemporalCentralDifference(dim),
            nn.BatchNorm3d(dim),
        )
        self.value = nn.Conv3d(dim, dim, kernel_size=1, bias=False)
        self.output = nn.Conv3d(dim, dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        _, _, _, height, width = x.shape
        region_size = (
            max(4 // self.temporal_patch, 1),
            max(height // 4, 1),
            max(width // 4, 1),
        )
        query, key, value = self.query(x), self.key(x), self.value(x)

        region_query = F.avg_pool3d(
            query.detach(),
            kernel_size=region_size,
            ceil_mode=True,
            count_include_pad=False,
        ).permute(0, 2, 3, 4, 1).flatten(1, 3)
        region_key = F.avg_pool3d(
            key.detach(),
            kernel_size=region_size,
            ceil_mode=True,
            count_include_pad=False,
        ).flatten(2, 4)
        routing_scores = region_query @ region_key
        topk = min(self.topk, routing_scores.shape[-1])
        region_graph = torch.topk(routing_scores, k=topk, dim=-1).indices
        region_graph = region_graph.unsqueeze(1).expand(
            -1, self.num_heads, -1, -1
        )

        output = regional_routing_attention(
            query,
            key,
            value,
            scale=self.scale,
            region_graph=region_graph,
            region_size=region_size,
        )
        output = output + self.local_context(value)
        return self.output(output)


class PeriodicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        temporal_patch: int,
        topk: int,
        mlp_ratio: float,
        side_kernel: int,
        drop_path: float,
    ):
        super().__init__()
        hidden_dim = int(mlp_ratio * dim)
        self.norm1 = nn.BatchNorm3d(dim)
        self.attention = PeriodicSparseAttention(
            dim, num_heads, temporal_patch, topk, side_kernel
        )
        self.norm2 = nn.BatchNorm3d(dim)
        self.mlp = nn.Sequential(
            nn.Conv3d(dim, hidden_dim, kernel_size=1),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, dim, kernel_size=1),
            nn.BatchNorm3d(dim),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.attention(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class TemporalPeriodicStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        temporal_patch: int,
        topk: int,
        mlp_ratio: float,
        side_kernel: int,
        drop_path: Sequence[float],
    ):
        super().__init__()
        if temporal_patch < 1 or temporal_patch & (temporal_patch - 1):
            raise ValueError("temporal_patch must be a positive power of two")
        self.temporal_patch = temporal_patch
        levels = int(math.log2(temporal_patch))
        self.downsample = nn.ModuleList(
            nn.Sequential(
                nn.BatchNorm3d(dim),
                nn.Conv3d(dim, dim, kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            )
            for _ in range(levels)
        )
        self.upsample = nn.ModuleList(
            nn.Sequential(
                nn.Upsample(scale_factor=(2, 1, 1), mode="nearest"),
                nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
                nn.BatchNorm3d(dim),
                nn.ELU(),
            )
            for _ in range(levels)
        )
        self.blocks = nn.ModuleList(
            PeriodicTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                temporal_patch=temporal_patch,
                topk=topk,
                mlp_ratio=mlp_ratio,
                side_kernel=side_kernel,
                drop_path=drop_path[index],
            )
            for index in range(depth)
        )

    def forward(self, x: Tensor) -> Tensor:
        original_frames = x.shape[2]
        padding = (-original_frames) % self.temporal_patch
        if padding:
            x = F.pad(x, (0, 0, 0, 0, 0, padding), mode="replicate")
        for layer in self.downsample:
            x = layer(x)
        for block in self.blocks:
            x = block(x)
        for layer in self.upsample:
            x = layer(x)
        return x[:, :, :original_frames]


class RhythmFormer(nn.Module):
    """Hierarchical temporal periodic transformer for RGB face videos."""

    def __init__(
        self,
        embed_dims: Sequence[int] | None = None,
        depths: Sequence[int] | None = None,
        temporal_patches: Sequence[int] | None = None,
        topks: Sequence[int] | None = None,
        mlp_ratios: Sequence[float] | None = None,
        head_dim: int | None = None,
        side_kernel: int | None = None,
        drop_path_rate: float | None = None,
        appearance_weight: float | None = None,
        difference_weight: float | None = None,
    ):
        super().__init__()
        embed_dims = tuple(embed_dims or config.rf_embed_dims)
        depths = tuple(depths or config.rf_depths)
        temporal_patches = tuple(temporal_patches or config.rf_temporal_patches)
        topks = tuple(topks or config.rf_topks)
        mlp_ratios = tuple(mlp_ratios or config.rf_mlp_ratios)
        head_dim = config.rf_head_dim if head_dim is None else head_dim
        side_kernel = config.rf_side_kernel if side_kernel is None else side_kernel
        drop_path_rate = (
            config.rf_drop_path_rate if drop_path_rate is None else drop_path_rate
        )
        appearance_weight = (
            config.rf_appearance_weight
            if appearance_weight is None
            else appearance_weight
        )
        difference_weight = (
            config.rf_difference_weight
            if difference_weight is None
            else difference_weight
        )

        stage_count = len(embed_dims)
        if not all(
            len(values) == stage_count
            for values in (depths, temporal_patches, topks, mlp_ratios)
        ):
            raise ValueError("all RhythmFormer stage parameter lists must match")
        if len(set(embed_dims)) != 1:
            raise ValueError(
                "the upstream RhythmFormer stages require a constant embedding width"
            )

        self.fusion_stem = FusionStem(appearance_weight, difference_weight)
        self.patch_embedding = nn.Conv3d(
            64, embed_dims[0], kernel_size=(1, 4, 4), stride=(1, 4, 4)
        )
        drop_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        stages = []
        offset = 0
        for index in range(stage_count):
            dim = embed_dims[index]
            if dim % head_dim:
                raise ValueError(
                    f"stage {index}: embed dim {dim} is not divisible by head dim {head_dim}"
                )
            stages.append(
                TemporalPeriodicStage(
                    dim=dim,
                    depth=depths[index],
                    num_heads=dim // head_dim,
                    temporal_patch=temporal_patches[index],
                    topk=topks[index],
                    mlp_ratio=mlp_ratios[index],
                    side_kernel=side_kernel,
                    drop_path=drop_rates[offset : offset + depths[index]],
                )
            )
            offset += depths[index]
        self.stages = nn.ModuleList(stages)
        self.head = nn.Conv1d(embed_dims[-1], 1, kernel_size=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x: Tensor) -> Tensor:
        batch, frames, _, _, _ = x.shape
        x = self.fusion_stem(x)
        spatial_height, spatial_width = x.shape[-2:]
        x = x.reshape(batch, frames, 64, spatial_height, spatial_width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.patch_embedding(x)
        if x.shape[-2] < 1 or x.shape[-1] < 1:
            raise ValueError("input face crop is too small for RhythmFormer")
        for stage in self.stages:
            x = stage(x)
        x = x.mean(dim=(-1, -2))
        return self.head(x).squeeze(1)
