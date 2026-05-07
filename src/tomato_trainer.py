"""
Custom trainer for TomatoPose-YOLO11-v3.

It keeps Ultralytics' original PoseTrainer behavior and only installs the
optional Tomato Geometry Consistency loss when lambda_geo > 0.
"""

from __future__ import annotations

from ultralytics.models.yolo.pose import PoseTrainer

from .tomato_loss import install_tomato_pose_loss


class TomatoPoseTrainer(PoseTrainer):
    """PoseTrainer with optional tomato geometry regularization."""

    geo_loss_gain: float = 0.0

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        if float(self.geo_loss_gain) > 0:
            install_tomato_pose_loss(model, gain=float(self.geo_loss_gain))
        return model
