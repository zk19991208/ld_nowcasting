"""Losses for quantitative radar-field reconstruction."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class RadarReconstructionLoss(nn.Module):
    """Weighted L1 with optional MSE and spatial-gradient penalties.

    Inputs are expected to be normalized to ``[0, 1]``. Thresholds are given
    in dBZ and converted with ``radar_vmax``. Pixel weights are selected from
    the target, so a bad prediction cannot lower its own weight.
    """

    def __init__(
        self,
        radar_vmax: float = 70.0,
        thresholds_dbz: Sequence[float] = (20.0, 35.0, 45.0),
        weights: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
        mse_weight: float = 0.2,
        gradient_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if radar_vmax <= 0:
            raise ValueError("radar_vmax must be positive")
        if len(weights) != len(thresholds_dbz) + 1:
            raise ValueError("weights must contain len(thresholds_dbz) + 1 values")
        if any(b <= a for a, b in zip(thresholds_dbz, thresholds_dbz[1:])):
            raise ValueError("thresholds_dbz must be strictly increasing")
        if any(v < 0 for v in weights):
            raise ValueError("reconstruction weights must be non-negative")

        self.radar_vmax = float(radar_vmax)
        self.mse_weight = float(mse_weight)
        self.gradient_weight = float(gradient_weight)
        self.register_buffer(
            "thresholds",
            torch.tensor(thresholds_dbz, dtype=torch.float32) / self.radar_vmax,
            persistent=False,
        )
        self.register_buffer(
            "region_weights", torch.tensor(weights, dtype=torch.float32), persistent=False
        )

    def _pixel_weights(self, target: torch.Tensor) -> torch.Tensor:
        # bucketize returns 0..len(thresholds), exactly matching region_weights.
        thresholds = self.thresholds.to(dtype=target.dtype)
        bins = torch.bucketize(target.detach().contiguous(), thresholds)
        return self.region_weights[bins]

    @staticmethod
    def _gradient_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
        true_dx = target[..., :, 1:] - target[..., :, :-1]
        pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
        true_dy = target[..., 1:, :] - target[..., :-1, :]
        return 0.5 * ((pred_dx - true_dx).abs().mean() + (pred_dy - true_dy).abs().mean())

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target shapes differ: {prediction.shape} vs {target.shape}"
            )
        pixel_weights = self._pixel_weights(target)
        weighted_l1 = (pixel_weights * (prediction - target).abs()).mean()
        mse = torch.mean(torch.square(prediction - target))
        gradient = self._gradient_l1(prediction, target)
        total = weighted_l1 + self.mse_weight * mse + self.gradient_weight * gradient
        parts = {
            "weighted_l1": weighted_l1,
            "mse": mse,
            "gradient": gradient,
            "total": total,
        }
        return total, parts
