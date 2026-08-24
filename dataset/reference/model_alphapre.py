"""AlphaPre — Amplitude–Phase Disentanglement for Precipitation Nowcasting.

Adapted from https://github.com/linkenghong/AlphaPre (CVPR 2025).
Architecture: AmpliNet (frequency-domain temporal mixing) + PhaseNet
  (phase evolution prediction) → AlphaMixer (spectral-mask fusion).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


# ==================================================================
#  Building blocks
# ==================================================================

class _Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, kernel_size=3,
                 padding_mode='zeros'):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size,
                              padding=kernel_size // 2,
                              padding_mode=padding_mode)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))


class _ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=8, kernel_size=3,
                 padding_mode='zeros'):
        super().__init__()
        self.block1 = _Block(dim, dim_out, groups, kernel_size, padding_mode)
        self.block2 = _Block(dim_out, dim_out, groups, kernel_size, padding_mode)
        self.res_conv = (nn.Conv2d(dim, dim_out, 1) if dim != dim_out
                         else nn.Identity())

    def forward(self, x):
        return self.block2(self.block1(x)) + self.res_conv(x)


# ==================================================================
#  AmpliNet — amplitude / intensity branch
# ==================================================================

class _AmpTimeCell(nn.Module):
    def __init__(self, t_in, t_out, size_factor=1):
        super().__init__()
        mid = int(t_out * size_factor)
        self.tmlp = nn.Sequential(nn.Linear(t_in, mid), nn.SELU(True),
                                  nn.Linear(mid, t_out))
        s = 0.02
        self.w1 = nn.Parameter(s * torch.randn(2, t_in, mid))
        self.b1 = nn.Parameter(s * torch.randn(2, 1, 1, 1, mid))
        self.w2 = nn.Parameter(s * torch.randn(2, mid, t_out))
        self.b2 = nn.Parameter(s * torch.randn(2, 1, 1, 1, t_out))

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        bias = self.tmlp(x)
        xf = torch.fft.rfft2(x, dim=[2, 3], norm="ortho")
        r1 = (torch.einsum('bchwt,to->bchwo', xf.real, self.w1[0]) -
              torch.einsum('bchwt,to->bchwo', xf.imag, self.w1[1]) + self.b1[0])
        i1 = (torch.einsum('bchwt,to->bchwo', xf.real, self.w1[1]) +
              torch.einsum('bchwt,to->bchwo', xf.imag, self.w1[0]) + self.b1[1])
        r1, i1 = F.relu(r1), F.relu(i1)
        r2 = (torch.einsum('bchwt,to->bchwo', r1, self.w2[0]) -
              torch.einsum('bchwt,to->bchwo', i1, self.w2[1]) + self.b2[0])
        i2 = (torch.einsum('bchwt,to->bchwo', r1, self.w2[1]) +
              torch.einsum('bchwt,to->bchwo', i1, self.w2[0]) + self.b2[1])
        x2 = torch.view_as_complex(torch.stack([r2, i2], dim=-1))
        x = torch.fft.irfft2(x2, dim=[2, 3], norm="ortho") + bias
        return x.permute(0, 4, 1, 2, 3)


class _AmpCell(nn.Module):
    def __init__(self, t_in, t_out, dim):
        super().__init__()
        self.t_out = t_out
        self.tmlp = nn.Sequential(nn.Linear(t_in, t_out), nn.SELU(True),
                                  nn.Linear(t_out, t_out))
        self.amptime = _AmpTimeCell(t_in, t_out)
        self.conv = nn.Sequential(
            nn.Conv2d(dim * t_out, dim * t_out, 3, padding=1),
            nn.GroupNorm(4, dim * t_out), nn.SiLU(),
            nn.Conv2d(dim * t_out, dim * t_out, 3, padding=1))

    def forward(self, x):
        res = self.tmlp(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        x = self.amptime(x) + res
        res = x
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        x = self.conv(x)
        x = rearrange(x, 'b (t c) h w -> b t c h w', t=self.t_out)
        return x + res


class _AmpliNet(nn.Module):
    def __init__(self, pre_seq, aft_seq, dim, hidden_dim, n_layers=3):
        super().__init__()
        self.pre, self.aft = pre_seq, aft_seq
        self.tmlp = nn.Sequential(nn.Linear(pre_seq, aft_seq * 2), nn.SELU(True),
                                  nn.Linear(aft_seq * 2, aft_seq))
        self.convin = nn.Sequential(_ResnetBlock(dim, hidden_dim),
                                    _ResnetBlock(hidden_dim, hidden_dim),
                                    nn.Conv2d(hidden_dim, hidden_dim, 1))
        self.cells = nn.ModuleList([
            _AmpCell(pre_seq if i == 0 else aft_seq, aft_seq, hidden_dim)
            for i in range(n_layers)])
        self.convout = nn.Sequential(_ResnetBlock(hidden_dim, hidden_dim),
                                     _ResnetBlock(hidden_dim, hidden_dim),
                                     nn.Conv2d(hidden_dim, dim, 1))

    def forward(self, x):
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.convin(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.pre)
        xr = self.tmlp(x.permute(0, 2, 3, 4, 1))
        xr = rearrange(xr, 'b c h w t -> (b t) c h w')
        for cell in self.cells:
            x = cell(x)
        x = xr + rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.convout(x)
        return rearrange(x, '(b t) c h w -> b t c h w', t=self.aft)


# ==================================================================
#  PhaseNet — phase evolution branch
# ==================================================================

class _PhaseNet(nn.Module):
    def __init__(self, input_shape, pre_seq, aft_seq, input_dim, hidden_dim):
        super().__init__()
        h, w = input_shape
        self.pre, self.aft = pre_seq, aft_seq
        tc_in = 2 + input_dim * pre_seq
        tc_out = input_dim * aft_seq
        self.pha_conv0 = nn.Conv2d(tc_in, tc_out, 1)
        self.phase_0 = nn.Sequential(
            _ResnetBlock(tc_in, hidden_dim, kernel_size=1),
            _ResnetBlock(hidden_dim, hidden_dim, kernel_size=1),
            nn.Conv2d(hidden_dim, tc_out, 1))
        self.phase_1 = nn.Sequential(
            _ResnetBlock(tc_in, hidden_dim, kernel_size=1),
            _ResnetBlock(hidden_dim, hidden_dim, kernel_size=1),
            nn.Conv2d(hidden_dim, tc_out, 1))
        self.phase_2 = nn.Sequential(
            _ResnetBlock(tc_in, hidden_dim, kernel_size=3, padding_mode='circular'),
            _ResnetBlock(hidden_dim, hidden_dim, kernel_size=3, padding_mode='circular'),
            nn.Conv2d(hidden_dim, tc_out, 1))
        self.pha_conv1 = nn.Conv2d(4 * tc_out, tc_out, 1)
        u = torch.fft.fftfreq(h)
        v = torch.fft.rfftfreq(w)
        u, v = torch.meshgrid(u, v, indexing='ij')
        self.register_buffer('uv', torch.stack((u, v), dim=0))

    def forward(self, x):
        B, T, C, H, W = x.shape
        xf = torch.fft.rfft2(x)
        x_amps = torch.abs(xf)
        x_phas = torch.angle(xf) / torch.pi
        xp = rearrange(x_phas, 'b t c h w -> b (t c) h w')
        xpuv = torch.cat([xp, self.uv.expand(B, -1, -1, -1)], 1)
        pt = self.pha_conv0(xpuv)
        p0 = pt + self.phase_0(xpuv)
        p1 = pt * self.phase_1(xpuv)
        p2 = pt * self.phase_2(xpuv)
        phas_t = self.pha_conv1(torch.cat([pt, p0, p1, p2], 1))
        phas_t = rearrange(phas_t, 'b (t c) h w -> b t c h w', t=self.aft)
        phas_t = x_phas[:, -1:] + phas_t
        phas_t_un = phas_t * torch.pi
        xf_out = x_amps[:, -1:] * torch.exp(1j * phas_t_un)
        return torch.fft.irfft2(xf_out), phas_t_un, x_amps


# ==================================================================
#  AlphaMixer — spectral-mask fusion
# ==================================================================

class _AlphaMixer(nn.Module):
    def __init__(self, input_shape, spec_num, input_dim, hidden_dim, aft_seq):
        super().__init__()
        h, w = input_shape
        self.aft = aft_seq
        mask = torch.zeros(h, w // 2 + 1)
        mask[:spec_num, :spec_num] = 1.0
        mask[-spec_num:, :spec_num] = 1.0
        self.register_buffer('spec_mask', mask)
        self.out_mixer = nn.Sequential(
            _ResnetBlock(3 * input_dim, hidden_dim),
            _ResnetBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, input_dim, 1))

    def forward(self, xas, xps, phas):
        xf = torch.fft.rfft2(xas)
        alpha_fft = torch.abs(xf) * self.spec_mask * torch.exp(1j * phas)
        alpha = torch.fft.irfft2(alpha_fft)
        xap = rearrange(torch.cat([xas, xps, alpha], 2),
                         'b t c h w -> (b t) c h w')
        return rearrange(self.out_mixer(xap),
                         '(b t) c h w -> b t c h w', t=self.aft)


# ==================================================================
#  AlphaPreNet — main interface for our project
# ==================================================================

class AlphaPreNet(nn.Module):
    """AlphaPre adapted for radar nowcast: (B, T_in*C, H, W) → (B, T_out*C, H, W)."""

    def __init__(self, in_frames: int, out_frames: int, n_vars: int = 1,
                 spatial_size: int = 256, hidden_dim: int = 64,
                 n_layers: int = 3, spec_num: int = 20):
        super().__init__()
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.n_vars = n_vars
        shape = (spatial_size, spatial_size)

        self.amplinet = _AmpliNet(in_frames, out_frames, n_vars, hidden_dim,
                                  n_layers)
        self.phasenet = _PhaseNet(shape, in_frames, out_frames, n_vars,
                                  hidden_dim)
        self.alphamixer = _AlphaMixer(shape, spec_num, n_vars, hidden_dim,
                                      out_frames)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x5d = x.reshape(B, self.in_frames, self.n_vars, H, W)

        xas = torch.sigmoid(self.amplinet(x5d))
        xps, phas_t, _ = self.phasenet(x5d)
        xt = self.alphamixer(xas, xps, phas_t)

        return xt.reshape(B, self.out_frames * self.n_vars, H, W)
