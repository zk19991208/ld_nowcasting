"""DiffCast — Residual Diffusion for Precipitation Nowcasting.

Faithful reimplementation from:
  Paper:  https://arxiv.org/abs/2312.06734  (CVPR 2024)
  Code:   https://github.com/DeminYu98/DiffCast

Components (all from the original code):
  1. SimVP backbone    — deterministic predictor  (μ)
  2. ContextNet        — global motion prior       (h)
  3. GTUNet            — temporal UNet denoiser
  4. GaussianDiffusion — residual diffusion framework

Adaptation for our project:
  - T_in=10 (input frames), T_out=20 (output frames), C=1 (dBZ)
  - Spatial: 256×256 (original uses 128×128)
  - Interface: (B, T*C, H, W)  ↔  internal (B, T, C, H, W)
  - Training: end-to-end (backbone + diffusion) with combined loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import namedtuple
from einops import rearrange
from einops.layers.torch import Rearrange

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])

# =====================================================================
# Helpers
# =====================================================================
def exists(x):
    return x is not None

def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)

def extract(a, t, x_shape):
    out = a.gather(-1, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

def normalize_to_neg_one(img):
    return img * 2 - 1

def unnormalize_to_zero_one(t):
    return (t + 1) * 0.5

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    return torch.linspace(scale * 0.0001, scale * 0.02, timesteps, dtype=torch.float64)

# =====================================================================
# SimVP Backbone (from models/simvp/simvp_iter.py)
# =====================================================================
class _BasicConv2d(nn.Module):
    def __init__(self, inc, outc, ks=3, s=1, p=0, d=1, up=False, act_norm=False):
        super().__init__()
        if up:
            self.conv = nn.Sequential(nn.Conv2d(inc, outc * 4, ks, 1, p, d),
                                      nn.PixelShuffle(2))
        else:
            self.conv = nn.Conv2d(inc, outc, ks, s, p, d)
        self.norm = nn.GroupNorm(2, outc)
        self.act = nn.SiLU()
        self.act_norm = act_norm

    def forward(self, x):
        y = self.conv(x)
        return self.act(self.norm(y)) if self.act_norm else y


class _ConvSC(nn.Module):
    def __init__(self, ci, co, ks=3, down=False, up=False, act_norm=True):
        super().__init__()
        stride = 2 if down else 1
        pad = (ks - stride + 1) // 2
        self.conv = _BasicConv2d(ci, co, ks, stride, pad, up=up, act_norm=act_norm)

    def forward(self, x):
        return self.conv(x)


class _GroupConv2d(nn.Module):
    def __init__(self, inc, outc, ks=3, s=1, p=0, g=1, act_norm=False):
        super().__init__()
        if inc % g != 0:
            g = 1
        self.conv = nn.Conv2d(inc, outc, ks, s, p, groups=g)
        self.norm = nn.GroupNorm(g, outc)
        self.act = nn.LeakyReLU(0.2, True)
        self.act_norm = act_norm

    def forward(self, x):
        y = self.conv(x)
        return self.act(self.norm(y)) if self.act_norm else y


class _gInceptionST(nn.Module):
    def __init__(self, ci, ch, co, incep_ker=(3, 5, 7, 11), groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(ci, ch, 1)
        self.layers = nn.ModuleList([
            _GroupConv2d(ch, co, k, 1, k // 2, groups, True) for k in incep_ker])

    def forward(self, x):
        x = self.conv1(x)
        return sum(l(x) for l in self.layers)


class _SimVPEncoder(nn.Module):
    def __init__(self, ci, ch, ns, ks):
        super().__init__()
        samplings = [False, True] * (ns // 2)
        samplings = samplings[:ns]
        self.enc = nn.Sequential(
            _ConvSC(ci, ch, ks, down=samplings[0]),
            *[_ConvSC(ch, ch, ks, down=s) for s in samplings[1:]])

    def forward(self, x):
        enc1 = self.enc[0](x)
        lat = enc1
        for i in range(1, len(self.enc)):
            lat = self.enc[i](lat)
        return lat, enc1


class _SimVPDecoder(nn.Module):
    def __init__(self, ch, co, ns, ks):
        super().__init__()
        samplings = [False, True] * (ns // 2)
        samplings = list(reversed(samplings[:ns]))
        self.dec = nn.Sequential(
            *[_ConvSC(ch, ch, ks, up=s) for s in samplings[:-1]],
            _ConvSC(ch, ch, ks, up=samplings[-1]))
        self.readout = nn.Conv2d(ch, co, 1)

    def forward(self, hid, enc1):
        for i in range(len(self.dec) - 1):
            hid = self.dec[i](hid)
        return self.readout(self.dec[-1](hid + enc1))


class _MidIncepNet(nn.Module):
    def __init__(self, ci, ch, n2, incep_ker=(3, 5, 7, 11), groups=8):
        super().__init__()
        self.N2 = n2
        enc = [_gInceptionST(ci, ch // 2, ch, incep_ker, groups)]
        for _ in range(1, n2):
            enc.append(_gInceptionST(ch, ch // 2, ch, incep_ker, groups))
        dec = [_gInceptionST(ch, ch // 2, ch, incep_ker, groups)]
        for _ in range(1, n2 - 1):
            dec.append(_gInceptionST(2 * ch, ch // 2, ch, incep_ker, groups))
        dec.append(_gInceptionST(2 * ch, ch // 2, ci, incep_ker, groups))
        self.enc = nn.ModuleList(enc)
        self.dec = nn.ModuleList(dec)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2 - 1:
                skips.append(z)
        z = self.dec[0](z)
        for i in range(1, self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], 1))
        return z.reshape(B, T, C, H, W)


class SimVPBackbone(nn.Module):
    """SimVP: iterative T_in → T_in prediction."""
    def __init__(self, C, H, W, T_in, T_out,
                 hid_S=64, hid_T=256, N_S=2, N_T=6):
        super().__init__()
        self.T_in, self.T_out = T_in, T_out
        rH = int(H / 2 ** (N_S / 2))
        rW = int(W / 2 ** (N_S / 2))
        self.enc = _SimVPEncoder(C, hid_S, N_S, 3)
        self.dec = _SimVPDecoder(hid_S, C, N_S, 3)
        self.hid = _MidIncepNet(T_in * hid_S, hid_T, N_T)
        self.mse = nn.MSELoss()

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.reshape(B * T, C, H, W)
        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape
        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z).reshape(B * T, C_, H_, W_)
        return self.dec(hid, skip).reshape(B, T, C, H, W)

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        preds = []
        cur = frames_in.clone()
        for _ in range(self.T_out // self.T_in):
            cur = self(cur)
            preds.append(cur)
        preds = torch.cat(preds, dim=1)
        loss = self.mse(preds, frames_gt) if compute_loss and frames_gt is not None else None
        return preds, loss


# =====================================================================
# ContextNet (GlobalNet) — from diffcast.py
# =====================================================================
class _ConvGRUCell(nn.Module):
    def __init__(self, idim, hdim, ks=3, n_layer=1):
        super().__init__()
        self.pad = ks // 2
        self.hdim = hdim
        self.n_layer = n_layer
        self.cur_states = [None] * n_layer
        self.conv_gates = nn.ModuleList([
            nn.Conv2d((idim if i == 0 else hdim) + hdim, 2 * hdim, ks, 1, self.pad)
            for i in range(n_layer)])
        self.conv_cans = nn.ModuleList([
            nn.Conv2d((idim if i == 0 else hdim) + hdim, hdim, ks, 1, self.pad)
            for i in range(n_layer)])

    def init_hidden(self, shape, device):
        b, _, h, w = shape
        self.cur_states = [torch.zeros(b, self.hdim, h, w, device=device)
                           for _ in range(self.n_layer)]

    def forward(self, x):
        for i in range(self.n_layer):
            h = self.cur_states[i]
            comb = torch.cat([x, h], 1)
            rg, ug = torch.sigmoid(self.conv_gates[i](comb)).chunk(2, 1)
            cand = torch.tanh(self.conv_cans[i](torch.cat([x, rg * h], 1)))
            h_next = (1 - ug) * h + ug * cand
            self.cur_states[i] = h_next
            x = h_next
        return x


class _DiffResBlock(nn.Module):
    """ResnetBlock without time embedding for ContextNet."""
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(dim, dim_out, 3, 1, 1),
                                nn.GroupNorm(groups, dim_out), nn.SiLU())
        self.b2 = nn.Sequential(nn.Conv2d(dim_out, dim_out, 3, 1, 1),
                                nn.GroupNorm(groups, dim_out), nn.SiLU())
        self.res = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        return self.b2(self.b1(x)) + self.res(x)


class ContextNet(nn.Module):
    def __init__(self, dim, dim_mults=(1, 2, 4, 8), channels=1):
        super().__init__()
        self.init_conv = nn.Conv2d(channels, dim, 7, padding=3)
        dims = [dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList()
        for i, (di, do) in enumerate(in_out):
            is_last = i >= len(in_out) - 1
            self.downs.append(nn.ModuleList([
                _DiffResBlock(di, di),
                _ConvGRUCell(di, di, 3, 1),
                nn.Sequential(
                    Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
                    nn.Conv2d(di * 4, do, 1)
                ) if not is_last else nn.Identity()
            ]))

    def _init_state(self, shape, device):
        for i, ml in enumerate(self.downs):
            s = list(shape)
            s[-2] //= 2 ** i
            s[-1] //= 2 ** i
            ml[1].init_hidden(s, device)

    def scan_ctx(self, frames):
        B, T, C, H, W = frames.shape
        self._init_state((B, C, H, W), frames.device)
        local_ctx = None
        global_ctx = None
        for i in range(T):
            x = self.init_conv(frames[:, i])
            ctx = []
            for resnet, gru, down in self.downs:
                x = gru(resnet(x))
                ctx.append(x)
                x = down(x)
            global_ctx = ctx
            if i == T // 2:
                local_ctx = [h.clone() for h in ctx]
        return global_ctx, local_ctx


# =====================================================================
# UNet building blocks (from diffcast.py)
# =====================================================================
class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)

class _SinPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), -1)

class _Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
    def forward(self, x, scale_shift=None):
        x = self.norm(self.proj(x))
        if exists(scale_shift):
            s, sh = scale_shift
            x = x * (s + 1) + sh
        return self.act(x)

class _ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)
                                 ) if exists(time_emb_dim) else None
        self.b1 = _Block(dim, dim_out, groups)
        self.b2 = _Block(dim_out, dim_out, groups)
        self.res = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    def forward(self, x, time_emb=None):
        ss = None
        if exists(self.mlp) and exists(time_emb):
            ss = rearrange(self.mlp(time_emb), 'b c -> b c 1 1').chunk(2, 1)
        return self.b2(self.b1(x, ss)) + self.res(x)


class _TempAttnModule(nn.Module):
    def __init__(self, dim, ks=21, dilation=3, reduction=16):
        super().__init__()
        dk = 2 * dilation - 1
        dp = (dk - 1) // 2
        ddk = ks // dilation + ((ks // dilation) % 2 - 1)
        ddp = dilation * (ddk - 1) // 2
        self.conv0 = nn.Conv2d(dim, dim, dk, padding=dp, groups=dim)
        self.conv_sp = nn.Conv2d(dim, dim, ddk, padding=ddp, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        red = max(dim // reduction, 4)
        self.fc = nn.Sequential(nn.Linear(dim, red, bias=False), nn.ReLU(True),
                                nn.Linear(red, dim, bias=False), nn.Sigmoid())
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        u = x.clone()
        a = self.conv1(self.conv_sp(self.conv0(x)))
        b, c, _, _ = x.shape
        se = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return se * a * u

class _TempAttn(nn.Module):
    def __init__(self, dim, ks=21):
        super().__init__()
        self.p1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.sg = _TempAttnModule(dim, ks)
        self.p2 = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        return self.p2(self.sg(self.act(self.p1(x)))) + x

class _Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        hd = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Conv2d(dim, hd * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hd, dim, 1)
    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = [rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads)
                    for t in self.to_qkv(x).chunk(3, 1)]
        sim = torch.einsum('b h d i, b h d j -> b h i j', q * self.scale, k)
        out = torch.einsum('b h i j, b h d j -> b h i d', sim.softmax(-1), v)
        return self.to_out(rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w))


def _Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1))

def _Upsample(dim, dim_out=None):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.Conv2d(dim, default(dim_out, dim), 3, padding=1))


# =====================================================================
# GTUNet — Global Temporal UNet (denoiser)
# =====================================================================
class GTUNet(nn.Module):
    def __init__(self, dim, T_in, dim_mults=(1, 2, 4, 8), groups=8):
        super().__init__()
        self.channels = T_in * 2
        self.out_dim = T_in
        self.random_or_learned_sinusoidal_cond = False

        init_dim = dim
        self.init_conv = nn.Conv2d(self.channels, init_dim, 7, padding=3)
        dims = [init_dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(_SinPosEmb(dim), nn.Linear(dim, time_dim),
                                      nn.GELU(), nn.Linear(time_dim, time_dim))
        self.frag_idx_mlp = nn.Sequential(_SinPosEmb(dim), nn.Linear(dim, time_dim),
                                          nn.GELU(), nn.Linear(time_dim, time_dim))

        block_klass = partial(_ResnetBlock, groups=groups)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i, (di, do) in enumerate(in_out):
            is_last = i >= len(in_out) - 1
            self.downs.append(nn.ModuleList([
                block_klass(di * 2, di, time_emb_dim=time_dim * 2),
                block_klass(di, di, time_emb_dim=time_dim * 2),
                nn.Sequential(_RMSNorm(di), _TempAttn(di)),
                _Downsample(di, do) if not is_last else nn.Conv2d(di, do, 3, padding=1)
            ]))
        mid = dims[-1]
        self.mid1 = block_klass(mid, mid, time_emb_dim=time_dim * 2)
        self.mid_attn = nn.Sequential(_RMSNorm(mid), _Attention(mid))
        self.mid2 = block_klass(mid, mid, time_emb_dim=time_dim * 2)

        for i, (di, do) in enumerate(reversed(in_out)):
            is_last = i == len(in_out) - 1
            self.ups.append(nn.ModuleList([
                block_klass(do + di, do, time_emb_dim=time_dim * 2),
                block_klass(do + di, do, time_emb_dim=time_dim * 2),
                nn.Sequential(_RMSNorm(do), _TempAttn(do)),
                _Upsample(do, di) if not is_last else nn.Conv2d(do, di, 3, padding=1)
            ]))
        self.final_res = block_klass(dim * 2, dim, time_emb_dim=time_dim * 2)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, cond=None, ctx=None, idx=None):
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        if exists(cond):
            cond = rearrange(cond, 'b t c h w -> b (t c) h w')
        cond = default(cond, lambda: torch.zeros_like(x))
        x = torch.cat((cond, x), 1)
        x = self.init_conv(x)
        r = x.clone()
        t = torch.cat((self.time_mlp(time), self.frag_idx_mlp(idx)), 1)
        h = []
        for lev_i, (b1, b2, attn, down) in enumerate(self.downs):
            x = b1(torch.cat((x, ctx[lev_i]), 1), t)
            h.append(x)
            x = attn(b2(x, t))
            h.append(x)
            x = down(x)
        x = self.mid2(self.mid_attn(self.mid1(x, t)), t)
        for b1, b2, attn, up in self.ups:
            x = b1(torch.cat((x, h.pop()), 1), t)
            x = attn(b2(torch.cat((x, h.pop()), 1), t))
            x = up(x)
        x = self.final_conv(self.final_res(torch.cat((x, r), 1), t))
        return rearrange(x, 'b (t c) h w -> b t c h w', t=self.out_dim)


# =====================================================================
# GaussianDiffusion with training support
# =====================================================================
class GaussianDiffusion(nn.Module):
    def __init__(self, model, ctx_net, timesteps=1000,
                 sampling_timesteps=250, objective='pred_v',
                 beta_schedule='linear'):
        super().__init__()
        self.model = model
        self.ctx_net = ctx_net
        self.objective = objective
        self.self_condition = False

        betas = linear_beta_schedule(timesteps)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, 0)
        ac_prev = F.pad(ac[:-1], (1, 0), value=1.0)

        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = sampling_timesteps
        self.is_ddim_sampling = sampling_timesteps < timesteps
        self.ddim_sampling_eta = 0.0

        reg = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        reg('betas', betas)
        reg('alphas_cumprod', ac)
        reg('alphas_cumprod_prev', ac_prev)
        reg('sqrt_ac', torch.sqrt(ac))
        reg('sqrt_one_minus_ac', torch.sqrt(1.0 - ac))
        reg('sqrt_recip_ac', torch.sqrt(1.0 / ac))
        reg('sqrt_recipm1_ac', torch.sqrt(1.0 / ac - 1))
        pv = betas * (1.0 - ac_prev) / (1.0 - ac)
        reg('post_var', pv)
        reg('post_log_var', torch.log(pv.clamp(min=1e-20)))
        reg('post_coef1', betas * torch.sqrt(ac_prev) / (1.0 - ac))
        reg('post_coef2', (1.0 - ac_prev) * torch.sqrt(alphas) / (1.0 - ac))

        snr = ac / (1 - ac)
        if objective == 'pred_v':
            reg('loss_weight', snr / (snr + 1))
        elif objective == 'pred_noise':
            reg('loss_weight', snr / snr)
        else:
            reg('loss_weight', snr)

    @property
    def device(self):
        return self.betas.device

    def load_backbone(self, backbone):
        self.backbone_net = backbone

    def _predict_start_from_v(self, x_t, t, v):
        return (extract(self.sqrt_ac, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_ac, t, x_t.shape) * v)

    def _predict_noise_from_start(self, x_t, t, x0):
        return ((extract(self.sqrt_recip_ac, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_ac, t, x_t.shape))

    def _predict_v(self, x0, t, noise):
        return (extract(self.sqrt_ac, t, x0.shape) * noise -
                extract(self.sqrt_one_minus_ac, t, x0.shape) * x0)

    def _q_sample(self, x0, t, noise):
        return (extract(self.sqrt_ac, t, x0.shape) * x0 +
                extract(self.sqrt_one_minus_ac, t, x0.shape) * noise)

    def model_predictions(self, x, t, cond=None, ctx=None, idx=None,
                          clip_x_start=False):
        out = self.model(x, t, cond=cond, ctx=ctx, idx=idx)
        clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else (lambda x: x)
        if self.objective == 'pred_v':
            x0 = clip(self._predict_start_from_v(x, t, out))
            noise = self._predict_noise_from_start(x, t, x0)
        elif self.objective == 'pred_noise':
            noise = out
            x0 = clip(extract(self.sqrt_recip_ac, t, x.shape) * x -
                       extract(self.sqrt_recipm1_ac, t, x.shape) * noise)
        else:
            x0 = clip(out)
            noise = self._predict_noise_from_start(x, t, x0)
        return ModelPrediction(noise, x0)

    @torch.no_grad()
    def ddim_sample(self, shape, cond=None, ctx=None, idx=None):
        B = shape[0]
        device = cond.device if cond is not None else self.device
        times = torch.linspace(-1, self.num_timesteps - 1,
                               self.sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        pairs = list(zip(times[:-1], times[1:]))
        x = torch.randn(shape, device=device)
        for t_now, t_next in pairs:
            tc = torch.full((B,), t_now, device=device, dtype=torch.long)
            noise, x0, *_ = self.model_predictions(x, tc, cond=cond, ctx=ctx,
                                                    idx=idx, clip_x_start=True)
            if t_next < 0:
                x = x0
                continue
            a = self.alphas_cumprod[t_now]
            a_next = self.alphas_cumprod[t_next]
            sigma = self.ddim_sampling_eta * ((1 - a / a_next) * (1 - a_next) / (1 - a)).sqrt()
            c = (1 - a_next - sigma ** 2).sqrt()
            x = x0 * a_next.sqrt() + c * noise + sigma * torch.randn_like(x)
        return x

    @torch.no_grad()
    def sample(self, frames_in, T_out):
        B, T_in, C, H, W = frames_in.shape
        backbone_out, _ = self.backbone_net.predict(frames_in)
        K = T_in

        fin = normalize_to_neg_one(frames_in)
        mu = normalize_to_neg_one(backbone_out)

        combined = torch.cat((fin, mu), dim=1)
        global_ctx, local_ctx = self.ctx_net.scan_ctx(combined)

        preds = []
        pre_frag = fin
        pre_mu = None
        n_frags = T_out // K
        for j in range(n_frags):
            mu_seg = mu[:, j * K:(j + 1) * K]
            cond = pre_frag - pre_mu if pre_mu is not None else torch.zeros_like(pre_frag)
            ctx = global_ctx if j > 0 else local_ctx
            y = self.ddim_sample(
                (B, K, C, H, W), cond=cond, ctx=ctx,
                idx=torch.full((B,), j, device=frames_in.device, dtype=torch.long))
            frag = y + mu_seg
            preds.append(frag)
            pre_frag = frag
            pre_mu = mu_seg

        out = unnormalize_to_zero_one(torch.cat(preds, 1)).clamp(0, 1)
        return out, backbone_out

    # ------------ Training (not in original repo) ------------------
    def compute_train_loss(self, frames_in, frames_gt, loss_alpha=0.5):
        """End-to-end training: L = α·L_diff + (1-α)·L_backbone."""
        B, T_in_actual, C, H, W = frames_in.shape
        K = T_in_actual
        T_out = frames_gt.shape[1]

        backbone_out, backbone_loss = self.backbone_net.predict(
            frames_in, frames_gt, compute_loss=True)

        fin = normalize_to_neg_one(frames_in)
        mu = normalize_to_neg_one(backbone_out)
        gt = normalize_to_neg_one(frames_gt)

        residual = gt - mu

        combined = torch.cat((fin, mu), dim=1)
        global_ctx, local_ctx = self.ctx_net.scan_ctx(combined)

        n_frags = T_out // K
        diff_loss = 0.0
        for j in range(n_frags):
            seg_gt = residual[:, j * K:(j + 1) * K]
            if j == 0:
                seg_cond = torch.zeros_like(seg_gt)
            else:
                seg_cond = residual[:, (j - 1) * K:j * K]

            ctx = global_ctx if j > 0 else local_ctx
            t = torch.randint(0, self.num_timesteps, (B,), device=fin.device).long()
            noise = torch.randn_like(seg_gt)
            seg_noisy = self._q_sample(seg_gt, t, noise)

            model_out = self.model(seg_noisy, t, cond=seg_cond, ctx=ctx,
                                   idx=torch.full((B,), j, device=fin.device, dtype=torch.long))

            if self.objective == 'pred_v':
                target = self._predict_v(seg_gt, t, noise)
            elif self.objective == 'pred_noise':
                target = noise
            else:
                target = seg_gt

            w = extract(self.loss_weight, t, model_out.shape)
            diff_loss = diff_loss + (w * F.mse_loss(model_out, target, reduction='none')).mean()

        diff_loss = diff_loss / n_frags
        total = loss_alpha * diff_loss + (1 - loss_alpha) * backbone_loss
        return total


# =====================================================================
# Wrapper for integration with our Lightning module
# =====================================================================
class DiffCastNet(nn.Module):
    """Wraps DiffCast for (B, T_in*C, H, W) → (B, T_out*C, H, W) interface."""
    def __init__(self, in_frames=10, out_frames=20, n_vars=1,
                 spatial_size=256, dim=64, dim_mults=(1, 2, 4, 8),
                 diffusion_timesteps=1000, sampling_timesteps=250,
                 objective='pred_v', loss_alpha=0.5,
                 simvp_hid_S=64, simvp_hid_T=256, simvp_N_S=2, simvp_N_T=6):
        super().__init__()
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.n_vars = n_vars
        self.loss_alpha = loss_alpha

        backbone = SimVPBackbone(
            C=n_vars, H=spatial_size, W=spatial_size,
            T_in=in_frames, T_out=out_frames,
            hid_S=simvp_hid_S, hid_T=simvp_hid_T,
            N_S=simvp_N_S, N_T=simvp_N_T)

        unet = GTUNet(dim=dim, T_in=in_frames, dim_mults=dim_mults)
        ctx_net = ContextNet(dim=dim, dim_mults=dim_mults, channels=n_vars)

        self.diffusion = GaussianDiffusion(
            model=unet, ctx_net=ctx_net,
            timesteps=diffusion_timesteps,
            sampling_timesteps=sampling_timesteps,
            objective=objective)
        self.diffusion.load_backbone(backbone)

    def _to_btchw(self, x, T):
        B = x.shape[0]
        return x.reshape(B, T, self.n_vars, x.shape[-2], x.shape[-1])

    def forward(self, x):
        """Inference: (B, T_in*C, H, W) → (B, T_out*C, H, W)."""
        fin = self._to_btchw(x, self.in_frames)
        out, _ = self.diffusion.sample(fin, self.out_frames)
        return out.reshape(x.shape[0], self.out_frames * self.n_vars,
                           x.shape[-2], x.shape[-1])

    def compute_diffcast_loss(self, pred_flat, target_flat, global_step=None):
        """Combined end-to-end loss for training.

        NOTE: pred_flat is ignored; we recompute from scratch because
        DiffCast needs internal access to backbone output and residuals.
        This method is called from _shared_step when use_diffcast_loss=True.
        """
        B = target_flat.shape[0]
        H, W = target_flat.shape[-2], target_flat.shape[-1]
        inp = self._last_input
        fin = self._to_btchw(inp, self.in_frames)
        fgt = self._to_btchw(target_flat, self.out_frames)
        return self.diffusion.compute_train_loss(fin, fgt, self.loss_alpha)

    def forward_train(self, x, target):
        """Full training forward: returns loss directly."""
        self._last_input = x
        fin = self._to_btchw(x, self.in_frames)
        fgt = self._to_btchw(target, self.out_frames)
        return self.diffusion.compute_train_loss(fin, fgt, self.loss_alpha)
