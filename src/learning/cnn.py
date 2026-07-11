from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.layers(x))


class BidCostCNN(nn.Module):
    def __init__(
        self,
        scalar_dim: int,
        output_dim: int,
        image_channels: int = 2,
        hidden_dim: int = 96,
        dropout: float = 0.05,
        image_pool: str = "global",
        fusion: str = "concat",
        image_encoder: str = "simple",
    ) -> None:
        super().__init__()
        if image_pool not in {"global", "grid"}:
            raise ValueError("image_pool must be 'global' or 'grid'.")
        if fusion not in {"concat", "scalar_residual"}:
            raise ValueError("fusion must be 'concat' or 'scalar_residual'.")
        if image_encoder not in {"simple", "wide", "residual"}:
            raise ValueError("image_encoder must be 'simple', 'wide', or 'residual'.")
        self.fusion = fusion
        pool_size = (1, 1) if image_pool == "global" else (4, 4)
        if image_encoder == "simple":
            image_feature_channels = 48
            self.image_encoder = nn.Sequential(
                nn.Conv2d(image_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, image_feature_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Flatten(),
            )
        elif image_encoder == "wide":
            image_feature_channels = 72
            self.image_encoder = nn.Sequential(
                nn.Conv2d(image_channels, 24, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(24, 48, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(48, image_feature_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Flatten(),
            )
        else:
            image_feature_channels = 96
            self.image_encoder = nn.Sequential(
                ConvBlock(image_channels, 32),
                ResidualBlock(32),
                nn.MaxPool2d(2),
                ConvBlock(32, 64),
                ResidualBlock(64),
                nn.MaxPool2d(2),
                ConvBlock(64, image_feature_channels),
                ResidualBlock(image_feature_channels),
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Flatten(),
            )
        image_feature_dim = image_feature_channels * pool_size[0] * pool_size[1]
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, hidden_dim),
            nn.ReLU(),
        )
        if fusion == "concat":
            self.head = nn.Sequential(
                nn.Linear(image_feature_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.scalar_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            self.image_correction_head = nn.Sequential(
                nn.Linear(image_feature_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            nn.init.zeros_(self.image_correction_head[-1].weight)
            nn.init.zeros_(self.image_correction_head[-1].bias)

    def forward(self, scalar_x: torch.Tensor, image_x: torch.Tensor) -> torch.Tensor:
        image_feat = self.image_encoder(image_x)
        scalar_feat = self.scalar_encoder(scalar_x)
        fused = torch.cat([scalar_feat, image_feat], dim=1)
        if self.fusion == "concat":
            return self.head(fused)
        return self.scalar_head(scalar_feat) + self.image_correction_head(fused)
