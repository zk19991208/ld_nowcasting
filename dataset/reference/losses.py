"""Loss functions for radar echo extrapolation.

Sources:
  1) Cao et al. (2025) GRL — MSE, PM (probability-matching) losses.
  2) Yang & Yuan (2023) GRL — threshold-based balanced MSE/MAE,
     DWT-based multi-scale SSIM spatial loss, multi-scale temporal loss.
  3) Standard L1, weighted-L1, PM-L1, Dice losses.

All point-wise losses accept  (pred, target)  with shape (B, C, H, W).
Sequence-aware losses (spatial_ms_ssim, temporal) expect
(B, T*C, H, W)  and require ``n_output_frames`` / ``n_vars``.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================
#  1. Point-wise losses
# ===================================================================

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def weighted_mse_loss(
    pred: torch.Tensor, target: torch.Tensor, exponent: float = 2.0,
) -> torch.Tensor:
    weight = 1.0 + target.pow(exponent)
    return (weight * (pred - target).pow(2)).mean()


def weighted_l1_loss(
    pred: torch.Tensor, target: torch.Tensor, exponent: float = 2.0,
) -> torch.Tensor:
    weight = 1.0 + target.pow(exponent)
    return (weight * (pred - target).abs()).mean()


# ===================================================================
#  2. Threshold-based Balanced MSE / MAE  (Yang & Yuan 2023)
#     Adapted from CM4nowcast/utils/Loss_Assemble.py BMSELoss/BMAELoss
#     Thresholds for dBZ [0,70] normalised to [0,1]:
#       10 dBZ → 0.143,  20 dBZ → 0.286,  35 dBZ → 0.500,  50 dBZ → 0.714
#     Strong-echo pixels receive up to 30× weight.
# ===================================================================

_DEFAULT_B_THRESHOLDS = [0.143, 0.286, 0.500, 0.714]
_DEFAULT_B_WEIGHTS = [1.0, 2.0, 5.0, 10.0, 30.0]


def _threshold_weights(
    target: torch.Tensor, thresholds: list, weights: list,
) -> torch.Tensor:
    """Per-pixel weight lookup: N thresholds produce N+1 bins."""
    w = torch.full_like(target, float(weights[-1]))
    for i in range(len(thresholds) - 1, -1, -1):
        w = torch.where(target < thresholds[i], float(weights[i]), w)
    return w


def balanced_mse_loss(
    pred: torch.Tensor, target: torch.Tensor,
    thresholds: list = None, weights: list = None,
) -> torch.Tensor:
    thresholds = thresholds or _DEFAULT_B_THRESHOLDS
    weights = weights or _DEFAULT_B_WEIGHTS
    with torch.no_grad():
        w = _threshold_weights(target, thresholds, weights)
    return (w * (pred - target).pow(2)).mean()


def balanced_mae_loss(
    pred: torch.Tensor, target: torch.Tensor,
    thresholds: list = None, weights: list = None,
) -> torch.Tensor:
    thresholds = thresholds or _DEFAULT_B_THRESHOLDS
    weights = weights or _DEFAULT_B_WEIGHTS
    with torch.no_grad():
        w = _threshold_weights(target, thresholds, weights)
    return (w * (pred - target).abs()).mean()


# ===================================================================
#  3. Probability-matching losses  (Cao et al. 2025)
# ===================================================================

def pm_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    B = pred.shape[0]
    pred_sorted, _ = pred.reshape(B, -1).sort(dim=1)
    tgt_sorted, _ = target.reshape(B, -1).sort(dim=1)
    return F.mse_loss(pred_sorted, tgt_sorted)


def pm_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    B = pred.shape[0]
    pred_sorted, _ = pred.reshape(B, -1).sort(dim=1)
    tgt_sorted, _ = target.reshape(B, -1).sort(dim=1)
    return F.l1_loss(pred_sorted, tgt_sorted)


# ===================================================================
#  4. Dice loss  (soft, differentiable)
# ===================================================================

def dice_loss(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.0,
) -> torch.Tensor:
    if threshold > 0:
        k = 20.0
        pred_s = torch.sigmoid(k * (pred - threshold))
        tgt_s = torch.sigmoid(k * (target - threshold))
    else:
        pred_s = pred
        tgt_s = target
    flat_p = pred_s.reshape(-1)
    flat_t = tgt_s.reshape(-1)
    intersection = (flat_p * flat_t).sum()
    return 1.0 - (2.0 * intersection + 1.0) / (flat_p.sum() + flat_t.sum() + 1.0)


# ===================================================================
#  5. SSIM / MS-SSIM helpers  (Yang & Yuan 2023, spatial loss)
# ===================================================================

def _fspecial_gauss_2d(size: int = 11, sigma: float = 1.5,
                       device: torch.device = None) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = torch.outer(g, g)
    return (g / g.sum()).unsqueeze(0).unsqueeze(0)


def _ssim_value(
    x: torch.Tensor, y: torch.Tensor, win: torch.Tensor,
    C1: float = 0.01 ** 2, C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Compute mean SSIM (scalar) over all spatial positions and channels."""
    pad = win.shape[-1] // 2
    nch = x.shape[1]
    w = win.expand(nch, -1, -1, -1)
    mu_x = F.conv2d(x, w, padding=pad, groups=nch)
    mu_y = F.conv2d(y, w, padding=pad, groups=nch)
    sigma_xx = F.conv2d(x * x, w, padding=pad, groups=nch) - mu_x * mu_x
    sigma_yy = F.conv2d(y * y, w, padding=pad, groups=nch) - mu_y * mu_y
    sigma_xy = F.conv2d(x * y, w, padding=pad, groups=nch) - mu_x * mu_y
    ssim_map = ((2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_xx + sigma_yy + C2))
    return ssim_map.mean()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - SSIM  (differentiable loss)."""
    win = _fspecial_gauss_2d(11, 1.5, device=pred.device)
    return 1.0 - _ssim_value(pred, target, win)


# ===================================================================
#  6. DWT-based multi-scale SSIM spatial loss  (Yang & Yuan 2023)
#     Adapted from CM4nowcast/utils/wsloss1/wavelet_ssim_loss.py
#     WSloss_linear_add_adhoc
# ===================================================================

def _haar_dwt2(x: torch.Tensor):
    """Single-level 2-D Haar DWT → (LL, LH, HL, HH)."""
    xL = (x[:, :, 0::2, :] + x[:, :, 1::2, :]) * 0.5
    xH = (x[:, :, 0::2, :] - x[:, :, 1::2, :]) * 0.5
    LL = (xL[:, :, :, 0::2] + xL[:, :, :, 1::2]) * 0.5
    LH = (xL[:, :, :, 0::2] - xL[:, :, :, 1::2]) * 0.5
    HL = (xH[:, :, :, 0::2] + xH[:, :, :, 1::2]) * 0.5
    HH = (xH[:, :, :, 0::2] - xH[:, :, :, 1::2]) * 0.5
    return LL, LH, HL, HH


def spatial_ms_ssim_loss(
    pred: torch.Tensor, target: torch.Tensor,
    n_levels: int = 3,
    weight_l: float = 1.0,
    weight_m: float = 1.0,
    weight_h: float = 1.0,
) -> torch.Tensor:
    """DWT + SSIM spatial loss (Yang & Yuan 2023).

    Computes 1-SSIM on the original scale, then recursively decomposes
    with Haar DWT and computes weighted (1-SSIM) on each subband:
      LL → weight_l,  LH/HL → weight_m,  HH → weight_h.
    """
    win = _fspecial_gauss_2d(11, 1.5, device=pred.device)
    loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    min_size = 11

    if pred.shape[-1] >= min_size and pred.shape[-2] >= min_size:
        loss = loss + (1.0 - _ssim_value(pred, target, win))

    p, t = pred, target
    for _ in range(n_levels):
        p_LL, p_LH, p_HL, p_HH = _haar_dwt2(p)
        t_LL, t_LH, t_HL, t_HH = _haar_dwt2(t)
        for p_sub, t_sub, ws in [
            (p_LH, t_LH, weight_m),
            (p_HL, t_HL, weight_m),
            (p_HH, t_HH, weight_h),
            (p_LL, t_LL, weight_l),
        ]:
            if p_sub.shape[-1] >= min_size and p_sub.shape[-2] >= min_size:
                loss = loss + ws * (1.0 - _ssim_value(p_sub, t_sub, win))
        p, t = p_LL, t_LL

    return loss


# ===================================================================
#  7. Multi-scale temporal consistency loss  (Yang & Yuan 2023)
#     Adapted from CM4nowcast/utils/multi_scale_temporal_loss.py
#     MTloss_add_linear — L1 on frame differences at multiple pool scales
# ===================================================================

def temporal_consistency_loss(
    pred: torch.Tensor, target: torch.Tensor,
    n_output_frames: int = 20, n_vars: int = 1,
    pool_kernels: list = None, scaler: float = 0.1,
) -> torch.Tensor:
    """L1 on consecutive frame differences + spatially pooled versions."""
    if pool_kernels is None:
        pool_kernels = [5, 7, 11]

    B, _, H, W = pred.shape
    T, C = n_output_frames, n_vars
    pred_5d = pred.reshape(B, T, C, H, W)
    tgt_5d = target.reshape(B, T, C, H, W)

    if T < 2:
        return torch.tensor(0.0, device=pred.device)

    pred_d = (pred_5d[:, 1:] - pred_5d[:, :-1]).reshape(-1, C, H, W)
    tgt_d = (tgt_5d[:, 1:] - tgt_5d[:, :-1]).reshape(-1, C, H, W)

    loss = F.l1_loss(pred_d, tgt_d) * scaler

    for k in pool_kernels:
        if H >= k and W >= k:
            pf_p = F.avg_pool2d(pred_d, k, stride=1, padding=0)
            pf_t = F.avg_pool2d(tgt_d, k, stride=1, padding=0)
            loss = loss + F.l1_loss(pf_p, pf_t) * scaler

    return loss


# ===================================================================
#  Registry & Factory
# ===================================================================

_REGISTRY = {
    "mse": mse_loss,
    "l1": l1_loss,
    "weighted_mse": weighted_mse_loss,
    "weighted_l1": weighted_l1_loss,
    "balanced_mse": balanced_mse_loss,
    "balanced_mae": balanced_mae_loss,
    "pm": pm_loss,
    "pm_l1": pm_l1_loss,
    "dice": dice_loss,
    "ssim": ssim_loss,
    "spatial_ms_ssim": spatial_ms_ssim_loss,
    "temporal": temporal_consistency_loss,
}


class NowcastLoss(nn.Module):
    """Composite loss assembled from config.

    Example configs::

        # Cao et al. 2025:  L = MSE + 10*PM
        loss:
          components:
            mse: {weight: 1.0}
            pm:  {weight: 10.0}

        # Yang & Yuan 2023 CM loss (fixed):
        loss:
          components:
            balanced_mse: {weight: 1.0}
            balanced_mae: {weight: 1.0}
            spatial_ms_ssim: {weight: 1.0, n_levels: 3}
            temporal: {weight: 1.0, n_output_frames: 20, n_vars: 1}
    """

    def __init__(self, loss_cfg: Dict):
        super().__init__()
        components = loss_cfg.get("components", {"mse": {"weight": 1.0}})
        self.parts = []
        for name, params in components.items():
            if name not in _REGISTRY:
                raise ValueError(
                    f"Unknown loss '{name}'. Available: {list(_REGISTRY.keys())}"
                )
            p = dict(params)
            w = float(p.pop("weight", 1.0))
            self.parts.append((name, _REGISTRY[name], w, p))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for name, fn, w, extra_kw in self.parts:
            total = total + w * fn(pred, target, **extra_kw)
        return total

    def __repr__(self) -> str:
        parts = [f"{name}(w={w})" for name, _, w, _ in self.parts]
        return f"NowcastLoss({' + '.join(parts)})"
