"""WADEPre v2 — Faithful reimplementation from the original repository.

Original: https://github.com/sonderlau/WADEPre  (KDD 2026)
All sub-modules are copied verbatim; only import paths and the
outer wrapper (WADEPreV2Net) are new.

Adaptation for T_in != T_out:
  A learned Conv2d(T_in, T_out, 1) projects the input time dimension
  so the main body always runs with T = T_out frames.
"""

import sys, os, torch, torch.nn as nn, torch.nn.functional as F
from typing import Dict, Literal, Tuple
from einops import rearrange, repeat
import ptwt

# =====================================================================
#  wavelet_transform  (verbatim from utils/wavelet_transform.py)
# =====================================================================
WaveletCoeffDict = Dict[str, torch.Tensor]

class WaveletTransform:
    def __init__(self, wavelet: str = "bior2.4", level: int = 3,
                 mode: str = "reflect"):
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

    def transform(self, input_tensor: torch.Tensor) -> WaveletCoeffDict:
        with torch.amp.autocast("cuda", enabled=False):
            coeffs = ptwt.wavedec2(input_tensor.float(),
                                   wavelet=self.wavelet,
                                   level=self.level, mode=self.mode)
        dt = input_tensor.dtype
        result: WaveletCoeffDict = {"A": coeffs[0].to(dt)}
        for l in range(1, self.level + 1):
            result[f"D{l}"] = torch.stack(coeffs[l], dim=2).to(dt)
        return result

    def reverse(self, inp: WaveletCoeffDict) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            coeffs_list = [inp["A"].float()]
            for l in range(1, self.level + 1):
                d = inp[f"D{l}"].float()
                H, V, D = torch.unbind(d, dim=2)
                coeffs_list.append((H, V, D))
            return ptwt.waverec2(coeffs_list,
                                 wavelet=self.wavelet).clone().contiguous()


# =====================================================================
#  zncc  (verbatim from utils/zncc.py)
# =====================================================================
def zncc(pred: torch.Tensor, truth: torch.Tensor, eps: float = 1e-8):
    assert pred.shape == truth.shape and pred.dim() == 4
    b, t, h, w = pred.shape
    x = pred.reshape(b * t, -1).float()
    y = truth.reshape(b * t, -1).float()
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    num = (x * y).sum(dim=1)
    denorm = (x.norm(dim=1) * y.norm(dim=1)).clamp_min(eps)
    rho = (num / denorm).clamp(-1.0, 1.0)
    return (0.5 * (1.0 - rho)).mean()


# =====================================================================
#  ResNet blocks  (verbatim from src/submodules/ResNet.py)
# =====================================================================
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, kernel_size=3,
                 padding_mode='zeros', group_norm=True):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size,
                              padding=kernel_size // 2,
                              padding_mode=padding_mode)
        self.norm = (nn.GroupNorm(groups, dim_out) if group_norm
                     else nn.BatchNorm2d(dim_out))
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=8, kernel_size=3,
                 padding_mode='zeros', dropout_rate: float = 0.1):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups, kernel_size, padding_mode)
        self.block2 = Block(dim_out, dim_out, groups, kernel_size, padding_mode)
        self.res_conv = (nn.Conv2d(dim, dim_out, 1) if dim != dim_out
                         else nn.Identity())
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = self.dropout(h)
        return h + self.res_conv(x)


class DilatedResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, dilation=1, groups=8):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(groups, dim_out), nn.GELU())
        self.block2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(groups, dim_out), nn.GELU())
        self.res_conv = (nn.Conv2d(dim, dim_out, 1) if dim != dim_out
                         else nn.Identity())
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = self.dropout(self.block1(x))
        h = self.block2(h)
        return h + self.res_conv(x)


# =====================================================================
#  FPN  (verbatim from src/submodules/FPN.py)
# =====================================================================
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return F.relu(self.bn3(self.conv3(out)) + self.shortcut(x))


class FPN(nn.Module):
    def __init__(self, num_blocks, block=Bottleneck, in_channels=3,
                 feature_channels=256, layer_channels=None, input_sizes=None):
        super().__init__()
        layer_channels = layer_channels or [64, 128, 256, 512]
        self.num_layers = len(num_blocks)
        self.input_adapters = nn.ModuleList()
        for ch in layer_channels:
            self.input_adapters.append(nn.Sequential(
                nn.Conv2d(in_channels, ch * block.expansion, 3, 1, 1, bias=False),
                nn.BatchNorm2d(ch * block.expansion), nn.ReLU(True)))
        self.bottom_up_layers = nn.ModuleList()
        for ch, nb in zip(layer_channels, num_blocks):
            self.in_planes = ch * block.expansion
            strides = [1] * nb
            layers = []
            for s in strides:
                layers.append(block(self.in_planes, ch, s))
                self.in_planes = ch * block.expansion
            self.bottom_up_layers.append(nn.Sequential(*layers))
        self.toplayer = nn.Conv2d(layer_channels[-1] * block.expansion,
                                  feature_channels, 1)
        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(layer_channels[self.num_layers - 2 - i] * block.expansion,
                      feature_channels, 1)
            for i in range(self.num_layers - 1)])
        self.smooth_layers = nn.ModuleList([
            nn.Conv2d(feature_channels, feature_channels, 3, 1, 1)
            for _ in range(self.num_layers - 1)])

    def forward(self, inputs):
        bu = [layer(ada(inp)) for inp, ada, layer in
              zip(inputs, self.input_adapters, self.bottom_up_layers)]
        p = self.toplayer(bu[-1])
        td = [p]
        for i in range(self.num_layers - 1):
            bi = self.num_layers - 2 - i
            y = self.lateral_layers[i](bu[bi])
            p = F.interpolate(p, size=y.shape[2:], mode='bilinear',
                              align_corners=False) + y
            td.append(p)
        td = td[::-1]
        return tuple(self.smooth_layers[i](td[i]) if i < self.num_layers - 1
                     else td[i] for i in range(self.num_layers))


# =====================================================================
#  ApproximationNetwork  (verbatim from src/Approximation.py)
# =====================================================================
class SpatioTemporalBlock(nn.Module):
    def __init__(self, dim, dilation, drop_rate=0.05):
        super().__init__()
        self.temporal_mlp = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(), nn.Conv1d(dim, dim, 1))
        self.dropout = nn.Dropout(drop_rate)
        self.spatial_mixer = DilatedResnetBlock(dim, dim, dilation=dilation)

    def forward(self, x):
        b, d, h, w = x.shape
        spatial = self.spatial_mixer(x)
        temporal = self.temporal_mlp(rearrange(spatial, "b d h w -> b d (h w)"))
        temporal = self.dropout(temporal)
        return x + rearrange(temporal, "b d (h w) -> b d h w", h=h, w=w)


class ApproximationNetwork(nn.Module):
    def __init__(self, hidden_size, timesteps, cell_numbers=3,
                 dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.time_steps = timesteps
        self.cell_numbers = cell_numbers
        self._built = False

    def _build(self, h, w):
        hs = self.hidden_size
        T = self.time_steps
        g = 32 if hs % 32 == 0 else (16 if hs % 16 == 0 else 8)
        self.encoder = nn.Conv2d(T, hs, 3, padding=1)
        self.temporal_injector = nn.Sequential(
            nn.Conv3d(1, hs // 4, (3, 3, 3), padding=(1, 1, 1)),
            nn.GELU(),
            nn.Conv3d(hs // 4, hs // 4, (T, 1, 1)))
        self.fusion = nn.Conv2d(hs // 4 + hs, hs, 1)
        self.norm_before = nn.GroupNorm(g, hs)
        self.mixer_layers = nn.Sequential(*[
            SpatioTemporalBlock(hs, 2 ** i) for i in range(self.cell_numbers)])
        self.norm_after = nn.GroupNorm(g, hs)
        self.decoder = nn.Conv2d(hs, T, 1)
        nn.init.constant_(self.decoder.weight, 0)
        nn.init.constant_(self.decoder.bias, 0)
        self._built = True

    def forward(self, data, wavelet):
        coeffs = wavelet.transform(data)
        origin_a = coeffs["A"]
        B, T, H, W = origin_a.shape
        if not self._built:
            self._build(H, W)
            for m in self.modules():
                if m is not self:
                    m.to(origin_a.device)
        x = self.norm_before(self.encoder(origin_a))
        temporal = self.temporal_injector(origin_a.unsqueeze(1)).squeeze(2)
        x = self.fusion(torch.cat([x, temporal], dim=1))
        x = self.norm_after(self.mixer_layers(x))
        x = self.decoder(x)
        coeffs["A"] = x + origin_a
        for i in range(wavelet.level):
            d = coeffs[f"D{i+1}"][:, -1:].repeat(1, T, 1, 1, 1)
            coeffs[f"D{i+1}"] = d
        return wavelet.reverse(coeffs).contiguous(), coeffs


# =====================================================================
#  DetailNetwork  (verbatim from src/Detail.py)
# =====================================================================
class DetailNetwork(nn.Module):
    def __init__(self, fpn_time, idr_dim=32, feature_channel=128,
                 layer_channels=None, num_blocks=None, dropout_rate=0.1):
        super().__init__()
        self.fpn = None
        layer_channels = layer_channels or [64, 128, 256]
        num_blocks = num_blocks or [4, 4, 4]
        self.fpn_time = fpn_time
        self.feature_channel = feature_channel
        self.layer_channels = layer_channels
        self.num_blocks = num_blocks
        self.idr_dim = idr_dim
        self.dropout_rate = dropout_rate

    def _build(self, input_sizes):
        T = self.fpn_time
        self.fpn = FPN(num_blocks=self.num_blocks, in_channels=T,
                       feature_channels=T, layer_channels=self.layer_channels,
                       input_sizes=input_sizes)
        self.temporal_mlp_before = nn.Sequential(
            nn.Linear(T, self.feature_channel), nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_channel, T))
        self.temporal_mlp_after = nn.Sequential(
            nn.Linear(T, self.feature_channel), nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_channel, T))
        self.idr = nn.ModuleList()
        for _ in range(len(self.layer_channels)):
            self.idr.append(nn.Sequential(
                nn.Conv2d(T, self.idr_dim, 3, padding=1),
                nn.GroupNorm(4, self.idr_dim), nn.GELU(),
                nn.Conv2d(self.idr_dim, T, 3, padding=1)))

    def forward(self, data, wavelet):
        coeffs = wavelet.transform(data)
        batch, time, _, _ = coeffs["A"].shape
        level = wavelet.level
        if self.fpn is None:
            sizes = [(coeffs[f"D{i+1}"].shape[-2], coeffs[f"D{i+1}"].shape[-1])
                     for i in range(level)]
            self._build(sizes)
            for m in self.modules():
                if m is not self:
                    m.to(data.device)

        details = []
        for i in range(level):
            d = coeffs[f"D{i+1}"]
            b, t, c, h, w = d.shape
            d = rearrange(d, "b t c h w -> b c h w t")
            d = self.temporal_mlp_before(d)
            d = rearrange(d, "b c h w t -> (b c) t h w", c=3)
            details.append(d.contiguous())

        fpn_output = self.fpn(details)
        for i in range(level):
            fpn_out = fpn_output[i] + self.idr[i](details[i])
            bc, t, h, w = fpn_out.shape
            fpn_out = rearrange(fpn_out, "b t h w -> b h w t").contiguous()
            fpn_out = self.temporal_mlp_after(fpn_out)
            fpn_out = rearrange(fpn_out, "(b c) h w t -> b t c h w",
                                c=3, h=h, w=w, b=batch)
            coeffs[f"D{i+1}"] = fpn_out.contiguous()

        last_a = coeffs["A"][:, -1]
        coeffs["A"] = repeat(last_a, "b h w -> b t h w", t=time)
        return wavelet.reverse(coeffs).contiguous(), coeffs


# =====================================================================
#  Refiner  (verbatim from src/Refiner.py)
# =====================================================================
class Refiner(nn.Module):
    def __init__(self, time_steps, hidden_dim=64, dropout_rate=0.1):
        super().__init__()
        assert hidden_dim % time_steps == 0
        self.kinematic_coupling = ResnetBlock(
            2 * time_steps, hidden_dim, dropout_rate=dropout_rate)
        self.kc_norm = nn.GroupNorm(min(32, time_steps), hidden_dim)
        self.drift_corrector = ResnetBlock(
            2 * time_steps, hidden_dim, dropout_rate=dropout_rate)
        self.drift_norm = nn.GroupNorm(min(32, time_steps), hidden_dim)
        mixer_channels = 2 * time_steps + 2 * hidden_dim
        groups = time_steps
        self.mixer = nn.Sequential(
            nn.GroupNorm(groups, mixer_channels),
            ResnetBlock(mixer_channels, hidden_dim,
                        dropout_rate=dropout_rate, groups=groups),
            nn.GroupNorm(groups, hidden_dim),
            ResnetBlock(hidden_dim, hidden_dim,
                        dropout_rate=dropout_rate, groups=groups))
        self.out_conv = nn.Conv2d(hidden_dim, time_steps, 1)
        nn.init.constant_(self.out_conv.weight, 0)
        nn.init.constant_(self.out_conv.bias, 0)

    def forward(self, A_guide, D_guide, AD_guide, last_frame):
        B, T, H, W = A_guide.shape
        last_frame = repeat(last_frame, "b 1 h w -> b t h w", t=T)
        ad = torch.cat([A_guide, D_guide], dim=1)
        AD_interact = self.kc_norm(self.kinematic_coupling(ad))
        drift = torch.cat([AD_guide, last_frame], dim=1)
        drift = self.drift_norm(self.drift_corrector(drift))
        mx = torch.cat([AD_guide, A_guide, AD_interact, drift], dim=1)
        return AD_guide + self.out_conv(self.mixer(mx))


# =====================================================================
#  WADEPreV2Net — faithful wrapper
# =====================================================================
class WADEPreV2Net(nn.Module):
    """Faithful WADEPre for (B, T_in*C, H, W) → (B, T_out*C, H, W).

    Uses the *exact* original sub-modules.  When T_in != T_out a learned
    Conv2d temporal projection is prepended.

    Also exposes `compute_wadepre_loss()` so the Lightning module can
    use the paper's multi-scale curriculum loss.
    """

    def __init__(
        self,
        in_frames: int = 10,
        out_frames: int = 20,
        n_vars: int = 1,
        spatial_size: int = 256,
        # original hyper-params (from train.py / hyperparameters.yaml)
        approx_hidden_size: int = 256,
        approx_cells: int = 3,
        detail_idr_dim: int = 64,
        detail_feature_channel: int = 128,
        detail_layer_channels: list = None,
        detail_num_blocks: int = 4,
        refine_hidden_dim: int = 320,   # must be divisible by out_frames
        wavelet_name: str = "bior2.4",
        wavelet_level: int = 3,
        dropout_rate: float = 0.1,
        # loss curriculum params
        loss_a_weight: float = 1.0,
        loss_a_constant_weight: float = 0.05,
        loss_a_stop_step: int = 5000,
        loss_d_weight: float = 0.05,
        loss_recon_mean_weight: float = 0.01,
    ):
        super().__init__()
        if detail_layer_channels is None:
            detail_layer_channels = [64, 128, 256]

        self.in_frames = in_frames
        self.out_frames = out_frames
        self.n_vars = n_vars
        T = out_frames

        self.need_proj = (in_frames != out_frames)
        if self.need_proj:
            self.time_proj = nn.Conv2d(in_frames * n_vars, T * n_vars, 1)

        self.wavelet_transform = WaveletTransform(wavelet_name, wavelet_level,
                                                  "reflect")
        nb = ([detail_num_blocks] * len(detail_layer_channels)
              if isinstance(detail_num_blocks, int) else detail_num_blocks)

        self.detail_network = DetailNetwork(
            fpn_time=T, idr_dim=detail_idr_dim,
            feature_channel=detail_feature_channel,
            layer_channels=detail_layer_channels,
            num_blocks=nb, dropout_rate=dropout_rate)

        self.approx_network = ApproximationNetwork(
            hidden_size=approx_hidden_size, timesteps=T,
            cell_numbers=approx_cells, dropout_rate=dropout_rate)

        self.refine_mixer = Refiner(time_steps=T, hidden_dim=refine_hidden_dim,
                                    dropout_rate=dropout_rate)

        # loss params
        self.loss_a_weight = loss_a_weight
        self.loss_a_decay = ((loss_a_weight - loss_a_constant_weight)
                             / max(loss_a_stop_step, 1))
        self.loss_a_constant_weight = loss_a_constant_weight
        self.loss_a_stop_step = loss_a_stop_step
        self.loss_d_weight = loss_d_weight
        self.loss_recon_mean_weight = loss_recon_mean_weight

        self._do_dummy_run(spatial_size, T)

    # ---- eager init (required for DDP) ----
    def _do_dummy_run(self, spatial_size, T):
        dummy = torch.randn(2, T, spatial_size, spatial_size)
        with torch.no_grad():
            self.detail_network(dummy, self.wavelet_transform)
            self.approx_network(dummy, self.wavelet_transform)

    # ---- forward ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.need_proj:
            x = self.time_proj(x)
        B = x.shape[0]
        T = self.out_frames
        x_seq = x.reshape(B, T, self.n_vars, x.shape[-2], x.shape[-1])[:, :, 0]

        d_rec, d_coeff = self.detail_network(x_seq, self.wavelet_transform)
        a_rec, a_coeff = self.approx_network(x_seq, self.wavelet_transform)

        ad_coeff: WaveletCoeffDict = {"A": a_coeff["A"]}
        for l in range(1, self.wavelet_transform.level + 1):
            ad_coeff[f"D{l}"] = d_coeff[f"D{l}"]
        ad_rec = self.wavelet_transform.reverse(ad_coeff)

        refined = self.refine_mixer(
            A_guide=a_rec, D_guide=d_rec, AD_guide=ad_rec,
            last_frame=x_seq[:, -1:])

        self._last_details = {
            "d_rec": d_rec, "a_rec": a_rec, "ad_rec": ad_rec,
            "d_coeff": d_coeff, "a_coeff": a_coeff,
            "refined_out": refined,
        }

        return refined.unsqueeze(2).reshape(
            B, T * self.n_vars, refined.shape[-2], refined.shape[-1])

    # ---- WADEPre multi-scale curriculum loss ----
    def compute_wadepre_loss(self, pred_flat, target_flat, global_step):
        """Full paper loss: L_pred + w(t)*L_A + λ_D*L_D + λ_Mixed*L_Mixed."""
        det = self._last_details
        B = pred_flat.shape[0]
        T = self.out_frames
        truth = target_flat.reshape(B, T, -1, pred_flat.shape[-2],
                                    pred_flat.shape[-1])[:, :, 0]

        truth_wave = self.wavelet_transform.transform(truth)

        if global_step < self.loss_a_stop_step:
            a_w = self.loss_a_weight - global_step * self.loss_a_decay
        else:
            a_w = self.loss_a_constant_weight

        l_pred = F.mse_loss(det["refined_out"], truth)
        l_a = zncc(det["a_coeff"]["A"], truth_wave["A"])

        l_d = 0.0
        for l in range(1, self.wavelet_transform.level + 1):
            l_d += (F.mse_loss(det["d_coeff"][f"D{l}"], truth_wave[f"D{l}"])
                    * (1.0 / (2 ** l)))

        recon_mean = (det["ad_rec"] + det["a_rec"] + det["d_rec"]) / 3
        l_mixed = F.mse_loss(recon_mean, truth)

        total = (l_pred + a_w * l_a + self.loss_d_weight * l_d
                 + self.loss_recon_mean_weight * l_mixed)
        return total
