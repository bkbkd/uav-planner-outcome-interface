from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AsymmetricSmoothL1Loss(nn.Module):
    """SmoothL1 loss with a larger weight for underestimation."""

    def __init__(self, beta: float = 1.0, under_weight: float = 2.0) -> None:
        super().__init__()
        if beta <= 0.0:
            raise ValueError("--huber-beta must be positive.")
        if under_weight <= 0.0:
            raise ValueError("--asym-under-weight must be positive.")
        self.beta = float(beta)
        self.under_weight = float(under_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = F.smooth_l1_loss(pred, target, beta=self.beta, reduction="none")
        weights = torch.where(pred < target, self.under_weight, 1.0)
        return torch.mean(base * weights)


class AsymmetricL1Loss(nn.Module):
    """L1 loss with a larger weight for underestimation."""

    def __init__(self, under_weight: float = 2.0) -> None:
        super().__init__()
        if under_weight <= 0.0:
            raise ValueError("--asym-under-weight must be positive.")
        self.under_weight = float(under_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = torch.abs(pred - target)
        weights = torch.where(pred < target, self.under_weight, 1.0)
        return torch.mean(base * weights)
