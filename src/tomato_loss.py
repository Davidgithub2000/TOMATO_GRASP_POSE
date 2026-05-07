"""
Experimental Tomato Geometry Consistency regularizer.

Important:
    This module is designed for Ultralytics 8.3.x pose training and is kept
    deliberately conservative. It wraps the default pose loss and adds a weak
    structural penalty on raw predicted 4-keypoint layouts when possible.

    If the prediction tensor format cannot be recognized, the wrapper safely
    falls back to the original Ultralytics loss instead of stopping training.

Geometry prior:
    k0 = calyx, k1 = bottom, k2 = left, k3 = right

    1. Center consistency:
       midpoint(calyx, bottom) should be close to midpoint(left, right).

    2. Axis orthogonality:
       calyx-bottom axis and left-right axis should be approximately orthogonal.

This is a weak auxiliary regularizer. Recommended lambda_geo: 0.03-0.05.
"""

from __future__ import annotations

from types import MethodType
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


def _extract_raw_pred_kpts(preds: Any, nkpt: int = 4, ndim: int = 3) -> Optional[torch.Tensor]:
    """Try to extract raw keypoint predictions.

    Expected compatible shape:
        [B, nkpt * ndim, N] or [B, N, nkpt * ndim]

    Returns:
        Tensor [B, N, nkpt, ndim] or None.
    """
    candidates = []

    if torch.is_tensor(preds):
        candidates.append(preds)
    elif isinstance(preds, (list, tuple)):
        for item in preds:
            if torch.is_tensor(item):
                candidates.append(item)
            elif isinstance(item, (list, tuple)):
                candidates.extend([x for x in item if torch.is_tensor(x)])

    total = nkpt * ndim
    for t in candidates:
        if t.ndim != 3:
            continue
        # B, C, N
        if t.shape[1] == total:
            return t.permute(0, 2, 1).contiguous().view(t.shape[0], t.shape[2], nkpt, ndim)
        # B, N, C
        if t.shape[2] == total:
            return t.contiguous().view(t.shape[0], t.shape[1], nkpt, ndim)

    return None


class TomatoGeometryRegularizer(nn.Module):
    """Weak four-keypoint geometry regularizer."""

    def __init__(self, nkpt: int = 4, ndim: int = 3, eps: float = 1e-6):
        super().__init__()
        self.nkpt = nkpt
        self.ndim = ndim
        self.eps = eps

    def forward(self, raw_kpts: torch.Tensor) -> torch.Tensor:
        """Compute geometry regularization on raw predicted keypoint layouts.

        Args:
            raw_kpts: [B, N, 4, 3] or [B, N, 4, 2]
        """
        if raw_kpts is None or raw_kpts.ndim != 4 or raw_kpts.shape[2] < 4:
            return torch.tensor(0.0, device=raw_kpts.device if torch.is_tensor(raw_kpts) else "cpu")

        xy = raw_kpts[..., :2]
        calyx = xy[:, :, 0, :]
        bottom = xy[:, :, 1, :]
        left = xy[:, :, 2, :]
        right = xy[:, :, 3, :]

        vertical = bottom - calyx
        horizontal = right - left

        c1 = 0.5 * (calyx + bottom)
        c2 = 0.5 * (left + right)

        # Normalize by predicted object axis scale to avoid excessive penalty.
        scale = (
            vertical.norm(dim=-1).detach()
            + horizontal.norm(dim=-1).detach()
            + self.eps
        )

        center_loss = ((c1 - c2).norm(dim=-1) / scale).clamp(max=10.0).mean()

        dot = (vertical * horizontal).sum(dim=-1).abs()
        denom = vertical.norm(dim=-1) * horizontal.norm(dim=-1) + self.eps
        ortho_loss = (dot / denom).clamp(max=1.0).mean()

        return center_loss + 0.2 * ortho_loss


class TomatoPoseLossWrapper:
    """Wrap Ultralytics pose loss with a weak geometry regularizer."""

    def __init__(self, base_loss: Any, gain: float = 0.03, nkpt: int = 4, ndim: int = 3):
        self.base_loss = base_loss
        self.gain = float(gain)
        self.reg = TomatoGeometryRegularizer(nkpt=nkpt, ndim=ndim)

    def __call__(self, preds: Any, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        loss, loss_items = self.base_loss(preds, batch)

        if self.gain <= 0:
            return loss, loss_items

        raw_kpts = _extract_raw_pred_kpts(preds, nkpt=4, ndim=3)
        if raw_kpts is None:
            # Safe fallback: do not break training when Ultralytics changes internals.
            return loss, loss_items

        geo_loss = self.reg(raw_kpts)
        if not torch.isfinite(geo_loss):
            return loss, loss_items

        return loss + self.gain * geo_loss, loss_items


def install_tomato_pose_loss(model: Any, gain: float = 0.03) -> Any:
    """Monkey-patch a PoseModel instance to use TomatoPoseLossWrapper.

    This is called inside TomatoPoseTrainer.get_model(), after Ultralytics has
    built the actual model object used by the trainer.
    """
    if gain <= 0:
        return model

    original_init_criterion = model.init_criterion

    def init_criterion_with_tomato_geometry(self):
        base = original_init_criterion()
        return TomatoPoseLossWrapper(base, gain=gain, nkpt=4, ndim=3)

    model.init_criterion = MethodType(init_criterion_with_tomato_geometry, model)
    return model
