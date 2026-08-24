"""UNet for radar echo extrapolation with optional MEFM attention module.

Architecture:
  - Base: 4-level encoder-decoder UNet
  - Optional: MEFM (Multi-scale Extraction and Fusion Module) from
    Yang & Yuan (2023, GRL, doi:10.1029/2023GL103979)
    — Pyramid Pooling Attention + Inter-Scale Attention placed between
    encoder and decoder.
"""

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from losses import NowcastLoss
from metrics import RadarMetrics

logger = logging.getLogger(__name__)


# ===================================================================
#  Basic building blocks
# ===================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ===================================================================
#  MEFM — Multi-scale Extraction and Fusion Module
#  (Yang & Yuan 2023, Sec. 3.2 + Supporting Info S4)
# ===================================================================

class PyramidPoolingAttention(nn.Module):
    """Single-scale attention with pyramid pooling in K/V.

    Inspired by P2T (Wu et al. 2022, TPAMI) and PVT (Wang et al. 2021):
    when the spatial resolution is large, Q is spatially reduced via
    adaptive average pooling so the attention matrix stays tractable.
    K/V are pooled to multiple scales via pyramid pooling.

    Args:
        dim:        channel dimension of the input feature map.
        num_heads:  number of attention heads.
        pool_ratios: pyramid pooling ratios for K/V.
        max_tokens: when H*W exceeds this, Q is spatially reduced.
    """

    def __init__(self, dim: int, num_heads: int = 4,
                 pool_ratios: Tuple[int, ...] = (1, 2, 4, 8),
                 max_tokens: int = 1024):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pool_ratios = pool_ratios
        self.max_tokens = max_tokens

        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def _compute_sr(self, H: int, W: int) -> int:
        """Spatial reduction factor so that Q has at most *max_tokens* tokens."""
        sr = 1
        while (H // sr) * (W // sr) > self.max_tokens:
            sr *= 2
        return sr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x  # keep 4-D for the residual path

        sr = self._compute_sr(H, W)
        if sr > 1:
            x_q = F.adaptive_avg_pool2d(x, (H // sr, W // sr))
        else:
            x_q = x
        Hq, Wq = x_q.shape[2], x_q.shape[3]
        Nq = Hq * Wq

        Q = self.q_proj(x_q.flatten(2).transpose(1, 2))
        Q = Q.reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        kv_tokens = []
        for r in self.pool_ratios:
            eff_r = r * sr
            ph, pw = max(1, H // eff_r), max(1, W // eff_r)
            pooled = F.adaptive_avg_pool2d(x, (ph, pw))
            kv_tokens.append(pooled.flatten(2).transpose(1, 2))
        kv_cat = torch.cat(kv_tokens, dim=1)

        KV = self.kv_proj(kv_cat).reshape(B, -1, 2, self.num_heads, self.head_dim)
        KV = KV.permute(2, 0, 3, 1, 4)
        K, V = KV[0], KV[1]

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, Nq, C)
        out = self.out_proj(out)

        # reshape to spatial, upsample if needed, then residual + norm
        out = out.transpose(1, 2).reshape(B, C, Hq, Wq)
        if sr > 1:
            out = F.interpolate(out, size=(H, W), mode='bilinear',
                                align_corners=False)
        out_flat = (out + residual).flatten(2).transpose(1, 2)
        out_flat = self.norm(out_flat)
        return out_flat.transpose(1, 2).reshape(B, C, H, W)


class FFN(nn.Module):
    """Feed-forward network used after attention."""

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        out = self.norm(self.net(x_flat) + x_flat)
        return out.transpose(1, 2).reshape(B, C, H, W)


class MultiScaleAttentionBlock(nn.Module):
    """Multi-scale attention + FFN for a single encoder level."""

    def __init__(self, dim: int, num_heads: int = 4,
                 pool_ratios: Tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        self.attn = PyramidPoolingAttention(dim, num_heads, pool_ratios)
        self.ffn = FFN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.attn(x))


class InterScaleAttention(nn.Module):
    """Cross-scale attention that enables information exchange between
    features at different encoder levels.

    Yang & Yuan (2023), Sec. 3.2 & Supporting Info S4.3.

    All features are spatially pooled to a fixed token grid (pool_size²)
    before self-attention so that memory stays bounded regardless of the
    input resolution.  After attention the result is upsampled back and
    added as a residual.
    """

    def __init__(self, dims: List[int], unified_dim: int = 0,
                 num_heads: int = 4, pool_size: int = 8):
        super().__init__()
        self.dims = dims
        self.udim = unified_dim or max(dims)
        self.pool_size = pool_size
        self.n_tok_per_level = pool_size * pool_size

        self.proj_in = nn.ModuleList(
            [nn.Linear(d, self.udim) for d in dims]
        )
        self.proj_out = nn.ModuleList(
            [nn.Linear(self.udim, d) for d in dims]
        )

        self.num_heads = num_heads
        self.head_dim = self.udim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(self.udim, self.udim * 3)
        self.out_proj = nn.Linear(self.udim, self.udim)
        self.norm1 = nn.LayerNorm(self.udim)
        self.ffn = nn.Sequential(
            nn.Linear(self.udim, self.udim * 4),
            nn.GELU(),
            nn.Linear(self.udim * 4, self.udim),
        )
        self.norm2 = nn.LayerNorm(self.udim)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        tokens_list = []
        shapes = []
        ps = self.pool_size
        for i, feat in enumerate(features):
            B, C, H, W = feat.shape
            shapes.append((B, C, H, W))
            pooled = F.adaptive_avg_pool2d(feat, (ps, ps))
            tok = pooled.flatten(2).transpose(1, 2)   # (B, ps², C)
            tok = self.proj_in[i](tok)                 # (B, ps², udim)
            tokens_list.append(tok)

        B = features[0].shape[0]
        ntok = self.n_tok_per_level
        x = torch.cat(tokens_list, dim=1)  # (B, n_levels*ps², udim)
        residual = x

        QKV = self.qkv(x).reshape(B, -1, 3, self.num_heads, self.head_dim)
        QKV = QKV.permute(2, 0, 3, 1, 4)
        Q, K, V = QKV[0], QKV[1], QKV[2]

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, -1, self.udim)
        out = self.out_proj(out)
        out = self.norm1(out + residual)
        out = self.norm2(self.ffn(out) + out)

        results = []
        for i, (b, c, h, w) in enumerate(shapes):
            tok = out[:, i * ntok:(i + 1) * ntok]
            tok = self.proj_out[i](tok)
            small = tok.transpose(1, 2).reshape(b, c, ps, ps)
            up = F.interpolate(small, size=(h, w), mode='bilinear',
                               align_corners=False)
            results.append(up + features[i])
        return results


class MEFM(nn.Module):
    """Multi-scale Extraction and Fusion Module.

    Placed between the UNet encoder and decoder to enhance multi-scale
    feature representations via pyramid pooling self-attention at each
    level, followed by cross-level interaction.

    Args:
        dims: channel dimensions for each encoder level [c, 2c, 4c, 8c].
        num_heads: attention heads per module.
        pool_ratios: pyramid pooling ratios for K/V.
    """

    def __init__(self, dims: List[int], num_heads: int = 4,
                 pool_ratios: Tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        self.ms_blocks = nn.ModuleList([
            MultiScaleAttentionBlock(d, num_heads, pool_ratios) for d in dims
        ])
        self.inter_scale = InterScaleAttention(dims, num_heads=num_heads)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        enhanced = [blk(f) for blk, f in zip(self.ms_blocks, features)]
        return self.inter_scale(enhanced)


# ===================================================================
#  UNet (with optional MEFM)
# ===================================================================

class UNet(nn.Module):
    """4-level UNet with optional MEFM between encoder and decoder.

    Args:
        in_channels:  n_input_frames * n_vars
        out_channels: n_output_frames * n_vars
        base_ch:      base channel width (doubled at each encoder level)
        use_mefm:     if True, insert MEFM between encoder and decoder
        mefm_heads:   number of attention heads for MEFM
        mefm_pool_ratios: pyramid pooling ratios for MEFM
    """

    def __init__(
        self, in_channels: int, out_channels: int, base_ch: int = 64,
        use_mefm: bool = False, mefm_heads: int = 4,
        mefm_pool_ratios: Tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        c = base_ch
        self.use_mefm = use_mefm

        # Encoder
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(c * 8, c * 16)

        # MEFM (optional)
        if use_mefm:
            dims = [c, c * 2, c * 4, c * 8]
            self.mefm = MEFM(dims, num_heads=mefm_heads,
                             pool_ratios=mefm_pool_ratios)
            logger.info("MEFM enabled with dims=%s, heads=%d", dims, mefm_heads)

        # Decoder
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = DoubleConv(c * 16, c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)

        self.head = nn.Conv2d(c, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        if self.use_mefm:
            e1, e2, e3, e4 = self.mefm([e1, e2, e3, e4])

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)


class RadarNowcastModule(L.LightningModule):
    """Lightning wrapper for radar nowcasting with selectable backbone.

    Supported model types (cfg["model"]["type"]):
      - "unet"     : 4-level UNet with optional MEFM
      - "wadepre"  : WADEPre (wavelet dual-branch extrapolation)
      - "alphapre" : AlphaPre (amplitude-phase disentanglement)

    Includes pixel-level CSI / POD / FAR evaluation at configurable dBZ
    thresholds, computed every validation and test epoch.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.save_hyperparameters(cfg)

        n_vars = len(cfg["data"]["variables"])
        n_input = cfg["data"]["n_input_frames"]
        n_output = cfg["data"]["n_output_frames"]
        model_cfg = cfg["model"]
        model_type = model_cfg.get("type", "unet")

        if model_type == "wadepre":
            from model_wadepre import WADEPreNet
            wp_cfg = model_cfg.get("wadepre", {})
            self.net = WADEPreNet(
                in_frames=n_input, out_frames=n_output, n_vars=n_vars,
                spatial_size=wp_cfg.get("spatial_size", 256),
                hidden_size=wp_cfg.get("hidden_size", 128),
                wavelet_level=wp_cfg.get("wavelet_level", 3),
                refine_hidden=wp_cfg.get("refine_hidden", 128),
            )
            logger.info("Model: WADEPre (in=%d, out=%d)", n_input, n_output)
        elif model_type == "wadepre_v2":
            from model_wadepre_v2 import WADEPreV2Net
            wp_cfg = model_cfg.get("wadepre_v2", {})
            self.net = WADEPreV2Net(
                in_frames=n_input, out_frames=n_output, n_vars=n_vars,
                spatial_size=wp_cfg.get("spatial_size", 256),
                approx_hidden_size=wp_cfg.get("approx_hidden_size", 256),
                approx_cells=wp_cfg.get("approx_cells", 3),
                detail_idr_dim=wp_cfg.get("detail_idr_dim", 64),
                detail_feature_channel=wp_cfg.get("detail_feature_channel", 128),
                detail_layer_channels=wp_cfg.get("detail_layer_channels", [64, 128, 256]),
                detail_num_blocks=wp_cfg.get("detail_num_blocks", 4),
                refine_hidden_dim=wp_cfg.get("refine_hidden_dim", 320),
                wavelet_level=wp_cfg.get("wavelet_level", 3),
                dropout_rate=wp_cfg.get("dropout_rate", 0.1),
                loss_a_weight=wp_cfg.get("loss_a_weight", 1.0),
                loss_a_constant_weight=wp_cfg.get("loss_a_constant_weight", 0.05),
                loss_a_stop_step=wp_cfg.get("loss_a_stop_step", 5000),
                loss_d_weight=wp_cfg.get("loss_d_weight", 0.05),
                loss_recon_mean_weight=wp_cfg.get("loss_recon_mean_weight", 0.01),
            )
            logger.info("Model: WADEPre-v2 faithful (in=%d, out=%d)", n_input, n_output)
        elif model_type == "alphapre":
            from model_alphapre import AlphaPreNet
            ap_cfg = model_cfg.get("alphapre", {})
            self.net = AlphaPreNet(
                in_frames=n_input, out_frames=n_output, n_vars=n_vars,
                spatial_size=ap_cfg.get("spatial_size", 256),
                hidden_dim=ap_cfg.get("hidden_dim", 64),
                n_layers=ap_cfg.get("n_layers", 3),
                spec_num=ap_cfg.get("spec_num", 20),
            )
            logger.info("Model: AlphaPre (in=%d, out=%d)", n_input, n_output)
        elif model_type == "diffcast":
            from model_diffcast import DiffCastNet
            dc_cfg = model_cfg.get("diffcast", {})
            self.net = DiffCastNet(
                in_frames=n_input, out_frames=n_output, n_vars=n_vars,
                spatial_size=dc_cfg.get("spatial_size", 256),
                dim=dc_cfg.get("dim", 64),
                dim_mults=tuple(dc_cfg.get("dim_mults", [1, 2, 4, 8])),
                diffusion_timesteps=dc_cfg.get("diffusion_timesteps", 1000),
                sampling_timesteps=dc_cfg.get("sampling_timesteps", 250),
                objective=dc_cfg.get("objective", "pred_v"),
                loss_alpha=dc_cfg.get("loss_alpha", 0.5),
                simvp_hid_S=dc_cfg.get("simvp_hid_S", 64),
                simvp_hid_T=dc_cfg.get("simvp_hid_T", 256),
                simvp_N_S=dc_cfg.get("simvp_N_S", 2),
                simvp_N_T=dc_cfg.get("simvp_N_T", 6),
            )
            logger.info("Model: DiffCast (in=%d, out=%d, dim=%d)",
                        n_input, n_output, dc_cfg.get("dim", 64))
        else:
            base_ch = model_cfg.get("base_channels", 64)
            mefm_cfg = model_cfg.get("mefm", {})
            use_mefm = mefm_cfg.get("enabled", False)
            mefm_heads = mefm_cfg.get("num_heads", 4)
            mefm_pool = tuple(mefm_cfg.get("pool_ratios", [1, 2, 4, 8]))
            self.net = UNet(
                in_channels=n_input * n_vars,
                out_channels=n_output * n_vars,
                base_ch=base_ch,
                use_mefm=use_mefm,
                mefm_heads=mefm_heads,
                mefm_pool_ratios=mefm_pool,
            )
            logger.info("Model: UNet (base_ch=%d, mefm=%s)", base_ch, use_mefm)
        self.lr = cfg["train"]["lr"]
        self.weight_decay = cfg["train"]["weight_decay"]

        loss_cfg = cfg.get("loss", {"components": {"mse": {"weight": 1.0}}})
        self.criterion = NowcastLoss(loss_cfg)
        logger.info("Loss function: %s", self.criterion)

        sched_cfg = cfg["train"].get("scheduler", {})
        self.sched_patience = int(sched_cfg.get("patience", 3))
        self.sched_factor = float(sched_cfg.get("factor", 0.1))
        self.sched_min_lr = float(sched_cfg.get("min_lr", 1e-6))

        eval_cfg = cfg.get("eval", {})
        thresholds = tuple(eval_cfg.get("thresholds_dbz", [20, 35, 40]))
        per_leadtime = eval_cfg.get("per_leadtime", False)

        metric_kwargs = dict(
            thresholds_dbz=thresholds,
            per_leadtime=per_leadtime,
            n_output_frames=n_output,
            n_vars=n_vars,
        )
        self.val_metrics = RadarMetrics(**metric_kwargs)
        self.test_metrics = RadarMetrics(**metric_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _shared_step(self, batch, stage: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inp, tgt = batch

        use_diffcast = (hasattr(self.net, 'forward_train')
                        and self.hparams.get("loss", {}).get(
                            "use_diffcast_loss", False))

        if use_diffcast:
            bb = self.net.diffusion.backbone_net
            B = inp.shape[0]
            fin = inp.reshape(B, self.net.in_frames, self.net.n_vars,
                              inp.shape[-2], inp.shape[-1])
            if stage == "train":
                loss = self.net.forward_train(inp, tgt)
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("NaN/Inf loss detected at step %d, "
                                   "replacing with zero", self.global_step)
                    loss = torch.zeros_like(loss, requires_grad=True)
            else:
                loss = torch.tensor(0.0, device=inp.device)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                pred_5d, _ = bb.predict(fin.float())
                pred = pred_5d.reshape(B, -1, inp.shape[-2], inp.shape[-1])
                if torch.isnan(pred).any():
                    logger.warning("NaN in backbone prediction at epoch %d",
                                   self.current_epoch)
                    pred = torch.zeros_like(pred)
                pred = pred.clamp(0, 1)
            if stage != "train":
                loss = self.criterion(pred, tgt)
        else:
            pred = self(inp)
            use_wadepre_loss = (hasattr(self.net, 'compute_wadepre_loss')
                                and self.hparams.get("loss", {}).get(
                                    "use_wadepre_loss", False))
            if use_wadepre_loss:
                loss = self.net.compute_wadepre_loss(pred, tgt, self.global_step)
            else:
                loss = self.criterion(pred, tgt)

        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        return loss, pred, tgt

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        _, pred, tgt = self._shared_step(batch, "val")
        self.val_metrics.update(pred, tgt)

    def _log_metric_dict(self, metric_dict: Dict, stage: str):
        """Log scalar metrics and print a summary line to the console."""
        scalars = {}
        for key, val in metric_dict.items():
            if val.dim() == 0:
                self.log(f"{stage}/{key}", val,
                         prog_bar=(stage == "val" and "CSI_20dBZ" in key),
                         sync_dist=True)
                scalars[key] = val.item()

        if self.global_rank == 0 and scalars:
            parts = [f"{k}={v:.4f}" for k, v in scalars.items()]
            logger.info("[Epoch %d] %s metrics: %s",
                        self.current_epoch, stage, "  ".join(parts))

    def on_validation_epoch_end(self):
        metric_dict = self.val_metrics.compute()
        self._log_metric_dict(metric_dict, "val")
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        _, pred, tgt = self._shared_step(batch, "test")
        self.test_metrics.update(pred, tgt)

    def on_test_epoch_end(self):
        metric_dict = self.test_metrics.compute()
        self._log_metric_dict(metric_dict, "test")
        self.test_metrics.reset()

    def configure_optimizers(self):
        opt_name = self.hparams.get("train", {}).get("optimizer", "adam")
        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.995))
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr,
                weight_decay=self.weight_decay)

        sched_type = self.hparams.get("train", {}).get("scheduler", {}).get(
            "type", "plateau")
        if sched_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=self.sched_factor,
                patience=self.sched_patience, min_lr=self.sched_min_lr)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
