"""
Custom lightweight modules for TomatoPose-YOLO11-v3.

These modules are intentionally designed as single-input, channel-preserving
modules so that they are compatible with Ultralytics parse_model without
patching the framework source code.

Modules:
    FeatureBalance: learnable scalar feature balancing after each fused scale.
    SimAM: parameter-free attention, used as v2 baseline attention.
    ECA: efficient channel attention with no explicit channel argument.
    TomatoCA: tomato coordinate attention lite; preserves horizontal/vertical
              positional cues for calyx/bottom/left/right keypoints.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureBalance(nn.Module):
    """Learnable scale factor for a fused feature map.

    This is a conservative, Ultralytics-compatible implementation of weighted
    feature balancing. It does not change the feature shape.
    """

    def __init__(self, init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.alpha.clamp(0.0, 3.0)


class SimAM(nn.Module):
    """Simple, parameter-free attention module.

    Reference idea: SimAM enhances features using an energy function without
    introducing convolution parameters. This implementation is compatible with
    2D feature maps.
    """

    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = float(e_lambda)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = max(h * w - 1, 1)
        x_minus_mu_square = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        y = x_minus_mu_square / (
            4 * (x_minus_mu_square.sum(dim=(2, 3), keepdim=True) / n + self.e_lambda)
        ) + 0.5
        return x * self.activation(y)


class ECA(nn.Module):
    """Efficient Channel Attention.

    Channel count is not required because ECA uses Conv1d over the channel
    descriptor. Kernel size should be odd.
    """

    def __init__(self, k_size: int = 3):
        super().__init__()
        if k_size % 2 == 0:
            raise ValueError("ECA k_size must be odd.")
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)  # B, C, 1, 1
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # B, 1, C
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))  # B, C, 1, 1
        return x * y.expand_as(x)


class TomatoCA(nn.Module):
    """Tomato Coordinate Attention Lite.

    This module separately summarizes features along width and height to keep
    direction-aware position cues. It is parameter-light and channel preserving.

    For tomato four-keypoint pose:
        - vertical context helps calyx vs bottom;
        - horizontal context helps left vs right.
    """

    def __init__(self, eps: float = 1e-6, strength: float = 1.0):
        super().__init__()
        self.eps = float(eps)
        self.strength = nn.Parameter(torch.tensor(float(strength)))

    @staticmethod
    def _norm01(t: torch.Tensor, eps: float) -> torch.Tensor:
        t_min = t.amin(dim=(2, 3), keepdim=True)
        t_max = t.amax(dim=(2, 3), keepdim=True)
        return (t - t_min) / (t_max - t_min + eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Height attention: B,C,H,1, retains calyx-bottom vertical information.
        h_context = x.mean(dim=3, keepdim=True)
        # Width attention: B,C,1,W, retains left-right horizontal information.
        w_context = x.mean(dim=2, keepdim=True)

        h_att = torch.sigmoid(2.0 * self._norm01(h_context, self.eps) - 1.0)
        w_att = torch.sigmoid(2.0 * self._norm01(w_context, self.eps) - 1.0)

        s = self.strength.clamp(0.0, 2.0)
        return x * (1.0 + s * h_att) * (1.0 + s * w_att) / 4.0


class IdentityAttention(nn.Module):
    """No-op module for architecture debugging."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
