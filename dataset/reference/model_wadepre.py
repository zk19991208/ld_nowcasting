"""WADEPre — Wavelet-based Dual-branch Extrapolation for Precipitation.

Adapted from https://github.com/sonderlau/WADEPre
Key change: all sub-modules are eagerly built in __init__ (required for DDP).
When T_in != T_out a Conv2d(T_in, T_out, 1) temporal projection is prepended
so the wavelet body always runs with T = T_out frames.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import ptwt
from typing import Dict, List, Tuple
from einops import rearrange, repeat


# ==================================================================
#  Wavelet helpers
# ==================================================================
WaveletCoeffDict = Dict[str, torch.Tensor]


class WaveletTransform:
    def __init__(self, wavelet="bior2.4", level=3, mode="reflect"):
        self.wavelet, self.level, self.mode = wavelet, level, mode

    def transform(self, x: torch.Tensor) -> WaveletCoeffDict:
        with torch.amp.autocast("cuda", enabled=False):
            coeffs = ptwt.wavedec2(x.float(), wavelet=self.wavelet,
                                   level=self.level, mode=self.mode)
        dtype = x.dtype
        result: WaveletCoeffDict = {"A": coeffs[0].to(dtype)}
        for l in range(1, self.level + 1):
            result[f"D{l}"] = torch.stack(coeffs[l], dim=2).to(dtype)
        return result

    def reverse(self, inp: WaveletCoeffDict) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            c = [inp["A"].float()]
            for l in range(1, self.level + 1):
                H, V, D = torch.unbind(inp[f"D{l}"].float(), dim=2)
                c.append((H, V, D))
            return ptwt.waverec2(c, wavelet=self.wavelet).contiguous()

    def get_coeff_sizes(self, spatial_size: int, T: int):
        """Return sizes of each coeff level for eager module construction."""
        dummy = torch.randn(1, T, spatial_size, spatial_size)
        coeffs = ptwt.wavedec2(dummy, wavelet=self.wavelet,
                               level=self.level, mode=self.mode)
        a_h, a_w = coeffs[0].shape[-2:]
        detail_sizes = []
        for l in range(1, self.level + 1):
            d = coeffs[l][0]
            detail_sizes.append((d.shape[-2], d.shape[-1]))
        return (a_h, a_w), detail_sizes


# ==================================================================
#  Basic blocks
# ==================================================================
def _gn(channels, preferred=8):
    for g in (preferred, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


class _Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, ks=3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, ks, padding=ks // 2)
        self.norm = _gn(dim_out, groups)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))


class _ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=8, dropout=0.1):
        super().__init__()
        self.b1 = _Block(dim, dim_out, groups)
        self.b2 = _Block(dim_out, dim_out, groups)
        self.skip = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.b2(self.b1(x))) + self.skip(x)


class _DilatedBlock(nn.Module):
    def __init__(self, dim, dim_out, dilation=1, groups=8):
        super().__init__()
        p = dilation
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=p, dilation=p),
            _gn(dim_out, groups), nn.GELU(),
            nn.Conv2d(dim_out, dim_out, 3, padding=p, dilation=p),
            _gn(dim_out, groups), nn.GELU())
        self.skip = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


# ==================================================================
#  FPN (lighter — use GroupNorm instead of BatchNorm for DDP safety)
# ==================================================================
class _FPNBlock(nn.Module):
    def __init__(self, in_ch, mid_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1), _gn(mid_ch), nn.ReLU(True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1), _gn(mid_ch), nn.ReLU(True))
        self.skip = nn.Conv2d(in_ch, mid_ch, 1) if in_ch != mid_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class _FPN(nn.Module):
    def __init__(self, n_levels, in_ch, feat_ch, layer_chs):
        super().__init__()
        self.n = n_levels
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, lc, 3, 1, 1), _gn(lc), nn.ReLU(True))
            for lc in layer_chs])
        self.layers = nn.ModuleList([_FPNBlock(lc, lc) for lc in layer_chs])
        self.top = nn.Conv2d(layer_chs[-1], feat_ch, 1)
        self.laterals = nn.ModuleList([
            nn.Conv2d(layer_chs[self.n - 2 - i], feat_ch, 1)
            for i in range(self.n - 1)])
        self.smooths = nn.ModuleList([
            nn.Conv2d(feat_ch, feat_ch, 3, 1, 1) for _ in range(self.n - 1)])

    def forward(self, inputs):
        bu = [self.layers[i](self.adapters[i](inputs[i])) for i in range(self.n)]
        p = self.top(bu[-1])
        td = [p]
        for i in range(self.n - 1):
            bi = self.n - 2 - i
            lat = self.laterals[i](bu[bi])
            p = F.interpolate(p, size=lat.shape[2:], mode='bilinear',
                              align_corners=False) + lat
            td.append(p)
        td = td[::-1]
        return tuple(self.smooths[i](td[i]) if i < self.n - 1 else td[i]
                     for i in range(self.n))


# ==================================================================
#  SpatioTemporalBlock  (for ApproximationNetwork)
# ==================================================================
class _STBlock(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        self.spatial = _DilatedBlock(dim, dim, dilation)
        self.temporal = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(), nn.Conv1d(dim, dim, 1))

    def forward(self, x):
        b, d, h, w = x.shape
        s = self.spatial(x)
        t = self.temporal(rearrange(s, 'b d h w -> b d (h w)'))
        return x + rearrange(t, 'b d (h w) -> b d h w', h=h)


# ==================================================================
#  ApproximationNetwork (eagerly built)
# ==================================================================
class _ApproxNet(nn.Module):
    def __init__(self, T: int, a_h: int, a_w: int, hidden: int = 128,
                 n_cells: int = 3):
        super().__init__()
        g = 8 if hidden % 8 == 0 else 4
        q = max(hidden // 4, 8)
        self.enc = nn.Conv2d(T, hidden, 3, padding=1)
        self.t_inj = nn.Sequential(
            nn.Conv3d(1, q, (3, 3, 3), padding=(1, 1, 1)), nn.GELU(),
            nn.Conv3d(q, q, (T, 1, 1)))
        self.fuse = nn.Conv2d(q + hidden, hidden, 1)
        self.norm_b = nn.GroupNorm(g, hidden)
        self.cells = nn.Sequential(*[_STBlock(hidden, 2 ** i)
                                     for i in range(n_cells)])
        self.norm_a = nn.GroupNorm(g, hidden)
        self.dec = nn.Conv2d(hidden, T, 1)
        nn.init.constant_(self.dec.weight, 0)
        nn.init.constant_(self.dec.bias, 0)
        self.T = T

    def forward(self, origin_a, wavelet_level):
        B, T, H, W = origin_a.shape
        x = self.norm_b(self.enc(origin_a))
        ti = self.t_inj(origin_a.unsqueeze(1)).squeeze(2)
        x = self.fuse(torch.cat([x, ti], 1))
        x = self.norm_a(self.cells(x))
        return self.dec(x) + origin_a


# ==================================================================
#  DetailNetwork (eagerly built)
# ==================================================================
class _DetailNet(nn.Module):
    def __init__(self, T: int, detail_sizes: List[Tuple[int, int]],
                 layer_chs=None, feat_ch=64, idr_dim=32):
        super().__init__()
        layer_chs = layer_chs or [32, 64, 128]
        n = len(detail_sizes)
        assert n == len(layer_chs)
        self.n = n
        self.fpn = _FPN(n, T, T, layer_chs)
        self.mlp_b = nn.Sequential(nn.Linear(T, feat_ch), nn.GELU(),
                                   nn.Linear(feat_ch, T))
        self.mlp_a = nn.Sequential(nn.Linear(T, feat_ch), nn.GELU(),
                                   nn.Linear(feat_ch, T))
        self.idr = nn.ModuleList([nn.Sequential(
            nn.Conv2d(T, idr_dim, 3, 1, 1), _gn(idr_dim), nn.GELU(),
            nn.Conv2d(idr_dim, T, 3, 1, 1)) for _ in range(n)])
        self.T = T

    def forward(self, coeffs: WaveletCoeffDict, B: int):
        T = self.T
        details_raw = []
        for i in range(self.n):
            d = coeffs[f"D{i+1}"]  # (B, T, 3, h, w)
            b, t, c, h, w = d.shape
            d = self.mlp_b(rearrange(d, 'b t c h w -> b c h w t'))
            d = rearrange(d, 'b c h w t -> (b c) t h w', c=3)
            details_raw.append(d.contiguous())

        fpn_out = self.fpn(details_raw)
        for i in range(self.n):
            fo = fpn_out[i] + self.idr[i](details_raw[i])
            bc, t, h, w = fo.shape
            fo = self.mlp_a(rearrange(fo, 'b t h w -> b h w t'))
            fo = rearrange(fo, '(b c) h w t -> b t c h w', c=3, h=h, w=w, b=B)
            coeffs[f"D{i+1}"] = fo.contiguous()
        return coeffs


# ==================================================================
#  Refiner (lighter: use hidden_dim proportional to T)
# ==================================================================
class _Refiner(nn.Module):
    def __init__(self, T: int, hidden: int = 64):
        super().__init__()
        g = 8 if hidden % 8 == 0 else 4
        self.kc = _ResnetBlock(2 * T, hidden, g)
        self.dc = _ResnetBlock(2 * T, hidden, g)
        mix_ch = 2 * T + 2 * hidden
        gm = 8 if mix_ch % 8 == 0 else 4
        self.mixer = nn.Sequential(
            _gn(mix_ch, gm),
            _ResnetBlock(mix_ch, hidden, g),
            _gn(hidden, g),
            _ResnetBlock(hidden, hidden, g))
        self.out = nn.Conv2d(hidden, T, 1)
        nn.init.constant_(self.out.weight, 0)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, a, d, ad, last):
        B, T, H, W = a.shape
        lf = repeat(last, 'b 1 h w -> b t h w', t=T)
        kc = self.kc(torch.cat([a, d], 1))
        dc = self.dc(torch.cat([ad, lf], 1))
        return ad + self.out(self.mixer(torch.cat([ad, a, kc, dc], 1)))


# ==================================================================
#  ZNCC loss helper
# ==================================================================
def zncc(pred, truth, eps=1e-8):
    b, t, h, w = pred.shape
    x = pred.reshape(b * t, -1).float()
    y = truth.reshape(b * t, -1).float()
    x, y = x - x.mean(1, True), y - y.mean(1, True)
    num = (x * y).sum(1)
    den = (x.norm(dim=1) * y.norm(dim=1)).clamp_min(eps)
    return (0.5 * (1.0 - (num / den).clamp(-1, 1))).mean()


# ==================================================================
#  WADEPreNet — main interface
# ==================================================================
class WADEPreNet(nn.Module):
    """WADEPre: (B, T_in*C, H, W) → (B, T_out*C, H, W)."""

    def __init__(self, in_frames=10, out_frames=20, n_vars=1,
                 spatial_size=256, hidden_size=128, wavelet_level=3,
                 refine_hidden=64):
        super().__init__()
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.n_vars = n_vars
        T = out_frames

        self.need_proj = (in_frames != out_frames)
        if self.need_proj:
            self.time_proj = nn.Conv2d(in_frames * n_vars, T * n_vars, 1)

        self.wavelet = WaveletTransform("bior2.4", wavelet_level, "reflect")
        (a_h, a_w), det_sizes = self.wavelet.get_coeff_sizes(spatial_size, T)

        self.detail_net = _DetailNet(T, det_sizes)
        self.approx_net = _ApproxNet(T, a_h, a_w, hidden_size)
        self.refiner = _Refiner(T, refine_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.need_proj:
            x = self.time_proj(x)
        B = x.shape[0]
        x_seq = x.reshape(B, self.out_frames, self.n_vars,
                           x.shape[-2], x.shape[-1])[:, :, 0]

        coeffs = self.wavelet.transform(x_seq)
        T = self.out_frames

        # Approximation branch
        pred_a = self.approx_net(coeffs["A"], self.wavelet.level)
        a_coeff = {**coeffs, "A": pred_a}
        for i in range(self.wavelet.level):
            d = a_coeff[f"D{i+1}"][:, -1:].repeat(1, T, 1, 1, 1)
            a_coeff[f"D{i+1}"] = d
        a_rec = self.wavelet.reverse(a_coeff)

        # Detail branch
        d_coeff = {k: v.clone() for k, v in coeffs.items()}
        d_coeff = self.detail_net(d_coeff, B)
        last_a = d_coeff["A"][:, -1]
        d_coeff["A"] = repeat(last_a, 'b h w -> b t h w', t=T)
        d_rec = self.wavelet.reverse(d_coeff)

        # Mixed reconstruction
        ad_coeff: WaveletCoeffDict = {"A": pred_a}
        for l in range(1, self.wavelet.level + 1):
            ad_coeff[f"D{l}"] = d_coeff[f"D{l}"]
        ad_rec = self.wavelet.reverse(ad_coeff)

        out = self.refiner(a_rec, d_rec, ad_rec, x_seq[:, -1:])
        return out.unsqueeze(2).reshape(B, self.out_frames * self.n_vars,
                                        out.shape[-2], out.shape[-1])
