# 多尺度频域损失：在多个空间尺度上做 FFT，对谱差做加权 L2 风格项并平均。
# 支持 pred/target 为 (B,T,C,H,W)（SimVP 等）或 (B,C,H,W)；内部对 5D 会展平为 (B*T,C,H,W) 再算 avg_pool2d/fft2。
# 与主损失组合：在 YAML 中设 freq_loss_weight>0，或在代码里用 BasePlusMultiScaleFrequency 包装任意 nn.Module 损失。

import torch
from torch import nn


class MultiScaleFrequencyLoss(nn.Module):
    def __init__(self, alpha=1.0, scales=None):
        super().__init__()
        if scales is None:
            scales = [1, 2, 4]
        self.alpha = alpha
        self.scales = list(scales)

    def compute_weight(self, mag2):
        """mag2 为 |pred_fft - target_fft|^2（实数、非负），避免对复数误用 real+imag 再 sqrt 导致负数开方 NaN。"""
        base = torch.sqrt(mag2.clamp(min=0.0) + 1e-12)
        weights = base.pow(self.alpha)
        weights.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        normalizer = weights.amax(dim=(-1, -2), keepdim=True)
        weights = weights / (normalizer + 1e-8)
        return weights.clamp(0, 1)

    def forward(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError(f"pred/target 形状须一致，当前 {pred.shape} vs {target.shape}")
        if pred.dim() == 5:
            b, t, c, h, w = pred.shape
            pred = pred.reshape(b * t, c, h, w)
            target = target.reshape(b * t, c, h, w)
        elif pred.dim() != 4:
            raise ValueError(f"仅支持 4D (B,C,H,W) 或 5D (B,T,C,H,W)，当前 dim={pred.dim()}")

        total_loss = 0.0
        for s in self.scales:
            if s > 1:
                pred_s = torch.nn.functional.avg_pool2d(pred, s)
                target_s = torch.nn.functional.avg_pool2d(target, s)
            else:
                pred_s = pred
                target_s = target

            pred_fft = torch.fft.fft2(pred_s, norm="ortho")
            target_fft = torch.fft.fft2(target_s, norm="ortho")
            z = pred_fft - target_fft
            # 谱域能量 |z|^2 = z.real^2 + z.imag^2，恒 >= 0；勿用 (z**2).real+(z**2).imag 再 sqrt（可出现负值 -> NaN）
            mag2 = z.real.pow(2) + z.imag.pow(2)
            weights = self.compute_weight(mag2)
            total_loss = total_loss + torch.mean(mag2 * weights)

        out = total_loss / len(self.scales)
        # 避免展平后标量与原始 batch 语义混淆：此处为整段序列上的标量损失，与 L1 全元素均值同一量级习惯
        return out


class BasePlusMultiScaleFrequency(nn.Module):
    """total = base_loss(pred, target) + freq_weight * MultiScaleFrequencyLoss(pred, target)"""

    def __init__(self, base_loss: nn.Module, freq_weight: float, alpha: float = 1.0, scales=None):
        super().__init__()
        if scales is None:
            scales = [1, 2, 4]
        self.base_loss = base_loss
        self.freq_weight = float(freq_weight)
        self.freq_loss = MultiScaleFrequencyLoss(alpha=alpha, scales=list(scales))

    def forward(self, pred, target):
        return self.base_loss(pred, target) + self.freq_weight * self.freq_loss(pred, target)
