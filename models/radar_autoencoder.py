"""Deterministic frame-wise ResNet autoencoder for radar reflectivity fields."""

from __future__ import annotations

from collections.abc import Sequence

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loss.radar_reconstruction import RadarReconstructionLoss
from utils import add_parser


def _valid_group_count(channels: int, requested: int) -> int:
    groups = min(int(requested), int(channels))
    while channels % groups != 0:
        groups -= 1
    return groups


class ResBlock(nn.Module):
    """Pre-activation residual block using GroupNorm and SiLU."""

    def __init__(self, in_channels: int, out_channels: int, norm_groups: int = 32) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_valid_group_count(in_channels, norm_groups), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_valid_group_count(out_channels, norm_groups), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class RadarEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        block_out_channels: Sequence[int],
        layers_per_block: int = 2,
        norm_groups: int = 32,
    ) -> None:
        super().__init__()
        channels = tuple(int(v) for v in block_out_channels)
        if not channels:
            raise ValueError("block_out_channels cannot be empty")
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, stage_channels in enumerate(channels):
            blocks = [ResBlock(stage_channels, stage_channels, norm_groups)]
            blocks.extend(
                ResBlock(stage_channels, stage_channels, norm_groups)
                for _ in range(int(layers_per_block) - 1)
            )
            self.stages.append(nn.Sequential(*blocks))
            if index < len(channels) - 1:
                self.downsamples.append(Downsample(stage_channels, channels[index + 1]))
        self.norm_out = nn.GroupNorm(_valid_group_count(channels[-1], norm_groups), channels[-1])
        self.conv_out = nn.Conv2d(channels[-1], latent_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
        return self.conv_out(F.silu(self.norm_out(x)))


class RadarDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        block_out_channels: Sequence[int],
        layers_per_block: int = 2,
        norm_groups: int = 32,
        output_activation: str = "none",
    ) -> None:
        super().__init__()
        channels = tuple(int(v) for v in block_out_channels)
        self.output_activation = str(output_activation).lower()
        if self.output_activation not in {"none", "sigmoid"}:
            raise ValueError("output_activation must be 'none' or 'sigmoid'")
        self.conv_in = nn.Conv2d(latent_channels, channels[-1], kernel_size=3, padding=1)
        self.stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        for index, stage_channels in enumerate(reversed_channels):
            blocks = [ResBlock(stage_channels, stage_channels, norm_groups)]
            blocks.extend(
                ResBlock(stage_channels, stage_channels, norm_groups)
                for _ in range(int(layers_per_block) - 1)
            )
            self.stages.append(nn.Sequential(*blocks))
            if index < len(reversed_channels) - 1:
                self.upsamples.append(Upsample(stage_channels, reversed_channels[index + 1]))
        self.norm_out = nn.GroupNorm(_valid_group_count(channels[0], norm_groups), channels[0])
        self.conv_out = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.conv_in(z)
        for index, stage in enumerate(self.stages):
            z = stage(z)
            if index < len(self.upsamples):
                z = self.upsamples[index](z)
        z = self.conv_out(F.silu(self.norm_out(z)))
        if self.output_activation == "sigmoid":
            z = torch.sigmoid(z)
        return z


class RadarAutoEncoder(nn.Module):
    """A spatial autoencoder with no encoder-to-decoder long skip connections."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_channels: int = 8,
        block_out_channels: Sequence[int] = (64, 128, 256, 256),
        layers_per_block: int = 2,
        norm_groups: int = 32,
        output_activation: str = "none",
    ) -> None:
        super().__init__()
        self.downsample_factor = 2 ** (len(tuple(block_out_channels)) - 1)
        self.encoder = RadarEncoder(
            in_channels, latent_channels, block_out_channels, layers_per_block, norm_groups
        )
        self.decoder = RadarDecoder(
            out_channels,
            latent_channels,
            block_out_channels,
            layers_per_block,
            norm_groups,
            output_activation,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"encode expects BCHW, received shape {tuple(x.shape)}")
        h, w = x.shape[-2:]
        if h % self.downsample_factor or w % self.downsample_factor:
            raise ValueError(
                f"H/W must be divisible by {self.downsample_factor}, received {(h, w)}"
            )
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError(f"decode expects BCHW, received shape {tuple(z.shape)}")
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class RadarAutoEncoderLit(pl.LightningModule):
    """Lightning wrapper that samples frames from existing radar sequence batches."""

    def __init__(
        self,
        height: int,
        width: int,
        learning_rate: float = 1e-4,
        ae_base_channels: int = 64,
        ae_channel_multipliers: Sequence[int] = (1, 2, 4, 4),
        ae_layers_per_block: int = 2,
        ae_latent_channels: int = 8,
        ae_norm_groups: int = 32,
        ae_output_activation: str = "none",
        ae_frame_source: str = "all",
        ae_frames_per_sample: int = 1,
        ae_radar_vmax: float = 70.0,
        ae_loss_thresholds_dbz: Sequence[float] = (20.0, 35.0, 45.0),
        ae_loss_weights: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
        ae_mse_weight: float = 0.2,
        ae_gradient_weight: float = 0.1,
        ae_metric_thresholds_dbz: Sequence[float] = (25.0, 35.0, 45.0),
        ae_log_images: int = 1,
        ae_num_log_images: int = 4,
        ae_lr_patience: int = 5,
        ae_lr_factor: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        multipliers = tuple(int(v) for v in ae_channel_multipliers)
        block_channels = tuple(int(ae_base_channels) * v for v in multipliers)
        self.net = RadarAutoEncoder(
            in_channels=1,
            out_channels=1,
            latent_channels=int(ae_latent_channels),
            block_out_channels=block_channels,
            layers_per_block=int(ae_layers_per_block),
            norm_groups=int(ae_norm_groups),
            output_activation=ae_output_activation,
        )
        if int(height) % self.net.downsample_factor or int(width) % self.net.downsample_factor:
            raise ValueError(
                f"height/width must be divisible by f={self.net.downsample_factor}: "
                f"received {(height, width)}"
            )
        if ae_frame_source not in {"all", "input", "target"}:
            raise ValueError("ae_frame_source must be all, input, or target")
        if int(ae_frames_per_sample) <= 0:
            raise ValueError("ae_frames_per_sample must be positive")
        if int(ae_num_log_images) <= 0:
            raise ValueError("ae_num_log_images must be positive")
        self.loss_fn = RadarReconstructionLoss(
            radar_vmax=ae_radar_vmax,
            thresholds_dbz=ae_loss_thresholds_dbz,
            weights=ae_loss_weights,
            mse_weight=ae_mse_weight,
            gradient_weight=ae_gradient_weight,
        )
        metric_thresholds = torch.tensor(ae_metric_thresholds_dbz, dtype=torch.float64)
        self.register_buffer(
            "metric_thresholds", metric_thresholds / float(ae_radar_vmax), persistent=False
        )
        n_metrics = len(metric_thresholds)
        for name in ("hits", "misses", "false_alarms", "predicted", "observed"):
            self.register_buffer(f"val_{name}", torch.zeros(n_metrics, dtype=torch.float64), persistent=False)

    def forward(self, frames_bchw: torch.Tensor) -> torch.Tensor:
        return self.net(frames_bchw)

    def encode(self, frames_bchw: torch.Tensor) -> torch.Tensor:
        return self.net.encode(frames_bchw)

    def decode(self, latent_bchw: torch.Tensor) -> torch.Tensor:
        return self.net.decode(latent_bchw)

    def _candidate_frames(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        seqs_x, seqs_y = batch
        source = self.hparams.ae_frame_source
        if source == "input":
            return seqs_x
        if source == "target":
            return seqs_y
        return torch.cat((seqs_x, seqs_y), dim=1)

    def _select_frames(
        self, batch: tuple[torch.Tensor, torch.Tensor], random_selection: bool
    ) -> torch.Tensor:
        seq = self._candidate_frames(batch).float()
        if seq.ndim != 5 or seq.shape[2] != 1:
            raise ValueError(f"expected radar batch BT1HW, received {tuple(seq.shape)}")
        batch_size, time_steps, channels, height, width = seq.shape
        count = min(int(self.hparams.ae_frames_per_sample), time_steps)
        if random_selection:
            indices = torch.randint(time_steps, (batch_size, count), device=seq.device)
        else:
            indices = torch.linspace(0, time_steps - 1, steps=count, device=seq.device).long()
            indices = indices.unsqueeze(0).expand(batch_size, -1)
        gather_index = indices[:, :, None, None, None].expand(-1, -1, channels, height, width)
        selected = torch.gather(seq, dim=1, index=gather_index)
        return selected.reshape(batch_size * count, channels, height, width).contiguous()

    def _shared_step(self, batch, random_selection: bool):
        target = self._select_frames(batch, random_selection=random_selection)
        prediction = self(target)
        loss, parts = self.loss_fn(prediction, target)
        return loss, parts, prediction, target

    def training_step(self, batch, batch_idx):
        loss, parts, _, _ = self._shared_step(batch, random_selection=True)
        self.log_dict(
            {f"train/{key}": value for key, value in parts.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch[0].shape[0],
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._reset_validation_counts()

    def validation_step(self, batch, batch_idx):
        loss, parts, prediction, target = self._shared_step(batch, random_selection=False)
        self.log_dict(
            {f"val/{key}": value for key, value in parts.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        self.log("valid_loss_fx", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        clipped = prediction.detach().clamp(0.0, 1.0)
        self._update_validation_counts(clipped, target.detach())
        if batch_idx == 0 and int(self.hparams.ae_log_images):
            self._log_reconstruction_images(clipped, target.detach())

    @torch.no_grad()
    def _log_reconstruction_images(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> None:
        if not self.trainer.is_global_zero or self.logger is None:
            return
        experiment = getattr(self.logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "add_images"):
            return
        count = min(int(self.hparams.ae_num_log_images), prediction.shape[0])
        pred = prediction[:count].float().cpu()
        truth = target[:count].float().clamp(0.0, 1.0).cpu()
        error = (pred - truth).abs()
        experiment.add_images("val/reconstruction", pred, self.global_step)
        experiment.add_images("val/target", truth, self.global_step)
        experiment.add_images("val/absolute_error", error, self.global_step)

    def _reset_validation_counts(self) -> None:
        for name in ("hits", "misses", "false_alarms", "predicted", "observed"):
            getattr(self, f"val_{name}").zero_()

    @torch.no_grad()
    def _update_validation_counts(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        for index, threshold in enumerate(self.metric_thresholds):
            pred_event = prediction >= threshold
            true_event = target >= threshold
            self.val_hits[index] += torch.logical_and(pred_event, true_event).sum()
            self.val_misses[index] += torch.logical_and(~pred_event, true_event).sum()
            self.val_false_alarms[index] += torch.logical_and(pred_event, ~true_event).sum()
            self.val_predicted[index] += pred_event.sum()
            self.val_observed[index] += true_event.sum()

    @staticmethod
    def _distributed_sum(value: torch.Tensor) -> torch.Tensor:
        value = value.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def on_validation_epoch_end(self) -> None:
        eps = torch.finfo(torch.float64).eps
        hits = self._distributed_sum(self.val_hits)
        misses = self._distributed_sum(self.val_misses)
        false_alarms = self._distributed_sum(self.val_false_alarms)
        predicted = self._distributed_sum(self.val_predicted)
        observed = self._distributed_sum(self.val_observed)
        for index, threshold in enumerate(self.hparams.ae_metric_thresholds_dbz):
            label = f"{float(threshold):g}dBZ"
            csi = hits[index] / (hits[index] + misses[index] + false_alarms[index] + eps)
            bias = predicted[index] / (observed[index] + eps)
            pod = hits[index] / (hits[index] + misses[index] + eps)
            far = false_alarms[index] / (hits[index] + false_alarms[index] + eps)
            self.log_dict(
                {
                    f"CSI_{label}_val": csi.float(),
                    f"BIAS_{label}_val": bias.float(),
                    f"POD_{label}_val": pod.float(),
                    f"FAR_{label}_val": far.float(),
                },
                prog_bar=threshold in (35, 45, 35.0, 45.0),
                sync_dist=False,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.hparams.learning_rate))
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=int(self.hparams.ae_lr_patience),
            factor=float(self.hparams.ae_lr_factor),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_loss_fx",
                "frequency": 1,
            },
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("RadarAutoEncoderLit")
        add_parser(parser)
        parser.add_argument("--ae_base_channels", type=int, default=64)
        parser.add_argument(
            "--ae_channel_multipliers", nargs="+", type=int, default=[1, 2, 4, 4]
        )
        parser.add_argument("--ae_layers_per_block", type=int, default=2)
        parser.add_argument("--ae_latent_channels", type=int, default=8)
        parser.add_argument("--ae_norm_groups", type=int, default=32)
        parser.add_argument("--ae_output_activation", choices=["none", "sigmoid"], default="none")
        parser.add_argument("--ae_frame_source", choices=["all", "input", "target"], default="all")
        parser.add_argument("--ae_frames_per_sample", type=int, default=1)
        parser.add_argument("--ae_radar_vmax", type=float, default=70.0)
        parser.add_argument(
            "--ae_loss_thresholds_dbz", nargs="+", type=float, default=[20.0, 35.0, 45.0]
        )
        parser.add_argument(
            "--ae_loss_weights", nargs="+", type=float, default=[1.0, 2.0, 4.0, 8.0]
        )
        parser.add_argument("--ae_mse_weight", type=float, default=0.2)
        parser.add_argument("--ae_gradient_weight", type=float, default=0.1)
        parser.add_argument(
            "--ae_metric_thresholds_dbz", nargs="+", type=float, default=[25.0, 35.0, 45.0]
        )
        parser.add_argument("--ae_log_images", type=int, default=1)
        parser.add_argument("--ae_num_log_images", type=int, default=4)
        parser.add_argument("--ae_lr_patience", type=int, default=5)
        parser.add_argument("--ae_lr_factor", type=float, default=0.5)
        return parent_parser
