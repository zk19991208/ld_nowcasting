"""Frozen-AE latent-space predictor for radar nowcasting."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loss.radar_reconstruction import RadarReconstructionLoss
from utils import add_parser

from .radar_autoencoder import RadarAutoEncoderLit
from .simvp_core import SimVPNet, stride_generator


class LatentSimVPPredictor(nn.Module):
    """Direct multi-step SimVP predictor, optionally with an ablation residual."""

    def __init__(
        self,
        sequence_length: int,
        latent_channels: int,
        latent_height: int,
        latent_width: int,
        hid_s: int = 32,
        hid_t: int = 256,
        n_s: int = 4,
        n_t: int = 4,
        incep_ker: Sequence[int] = (3, 5, 7, 11),
        groups: int = 8,
        residual: str = "none",
    ) -> None:
        super().__init__()
        if sequence_length <= 0 or latent_channels <= 0:
            raise ValueError("sequence_length and latent_channels must be positive")
        if residual not in {"last", "none"}:
            raise ValueError("residual must be 'last' or 'none'")
        n_down = sum(stride == 2 for stride in stride_generator(int(n_s)))
        factor = 2**n_down
        if latent_height % factor or latent_width % factor:
            raise ValueError(
                f"latent H/W must be divisible by predictor factor {factor}, "
                f"received {(latent_height, latent_width)}"
            )
        self.sequence_length = int(sequence_length)
        self.latent_channels = int(latent_channels)
        self.residual = residual
        self.net = SimVPNet(
            (sequence_length, latent_channels, latent_height, latent_width),
            hid_S=int(hid_s),
            hid_T=int(hid_t),
            N_S=int(n_s),
            N_T=int(n_t),
            incep_ker=list(incep_ker),
            groups=int(groups),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 5:
            raise ValueError(f"expected BTCHW latent history, received {tuple(history.shape)}")
        if history.shape[1] != self.sequence_length or history.shape[2] != self.latent_channels:
            raise ValueError(
                f"expected T/C={(self.sequence_length, self.latent_channels)}, "
                f"received {(history.shape[1], history.shape[2])}"
            )
        output = self.net(history)
        if self.residual == "last":
            output = history[:, -1:] + output
        return output


def _load_stats(
    path: str,
    latent_channels: int,
    mean_values: Sequence[float] | None,
    std_values: Sequence[float] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mean_values is not None or std_values is not None:
        if mean_values is None or std_values is None:
            raise ValueError("predictor_latent_mean and predictor_latent_std must be supplied together")
        mean = np.asarray(mean_values, dtype=np.float32)
        std = np.asarray(std_values, dtype=np.float32)
    else:
        if not path:
            raise ValueError(
                "predictor_latent_stats_path is required for a new training run; "
                "compute it with preprocess/compute_radar_ae_latent_stats.py"
            )
        stats_path = Path(path).expanduser()
        if not stats_path.is_file():
            raise FileNotFoundError(f"latent statistics file does not exist: {stats_path}")
        with np.load(stats_path) as stats:
            mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)
    if mean.shape != (latent_channels,) or std.shape != (latent_channels,):
        raise ValueError(
            f"latent stats must have shape ({latent_channels},), received {mean.shape}/{std.shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("latent mean/std contain invalid values")
    std = np.maximum(std, 1e-6)
    return torch.from_numpy(mean), torch.from_numpy(std)


class RadarLatentPredictorLit(pl.LightningModule):
    """Train a temporal predictor while keeping a pretrained radar AE frozen."""

    def __init__(
        self,
        height: int,
        width: int,
        input_length: int,
        target_length: int,
        predictor_ae_checkpoint: str,
        predictor_latent_stats_path: str = "",
        predictor_latent_mean: Sequence[float] | None = None,
        predictor_latent_std: Sequence[float] | None = None,
        predictor_hid_s: int = 32,
        predictor_hid_t: int = 256,
        predictor_n_s: int = 4,
        predictor_n_t: int = 4,
        predictor_incep_ker: Sequence[int] = (3, 5, 7, 11),
        predictor_groups: int = 8,
        predictor_residual: str = "none",
        predictor_encode_chunk_size: int = 4,
        predictor_decode_chunk_size: int = 4,
        predictor_huber_beta: float = 1.0,
        predictor_lambda_field: float = 0.0,
        predictor_field_frames_per_sample: int = 2,
        predictor_metric_thresholds_dbz: Sequence[float] = (25.0, 35.0, 45.0),
        predictor_metric_leads: Sequence[int] = (1, 5, 10, 15, 20),
        predictor_metric_height: int = 550,
        predictor_metric_width: int = 550,
        predictor_log_images: int = 1,
        predictor_num_log_images: int = 4,
        ae_radar_vmax: float = 70.0,
        ae_loss_thresholds_dbz: Sequence[float] = (20.0, 35.0, 45.0),
        ae_loss_weights: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
        ae_mse_weight: float = 0.2,
        ae_gradient_weight: float = 0.1,
        learning_rate: float = 2e-4,
        predictor_weight_decay: float = 1e-4,
        predictor_lr_patience: int = 5,
        predictor_lr_factor: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()
        if input_length != target_length:
            raise ValueError("first latent SimVP version requires input_length == target_length")
        if not predictor_ae_checkpoint:
            raise ValueError("predictor_ae_checkpoint is required")
        if predictor_encode_chunk_size <= 0 or predictor_decode_chunk_size <= 0:
            raise ValueError("encode/decode chunk sizes must be positive")
        if predictor_huber_beta <= 0:
            raise ValueError("predictor_huber_beta must be positive")
        if predictor_lambda_field < 0:
            raise ValueError("predictor_lambda_field must be non-negative")
        if predictor_field_frames_per_sample <= 0:
            raise ValueError("predictor_field_frames_per_sample must be positive")

        self.save_hyperparameters()
        self.ae = RadarAutoEncoderLit.load_from_checkpoint(
            predictor_ae_checkpoint,
            map_location="cpu",
        )
        self.ae.eval().requires_grad_(False)
        ae_height = int(self.ae.hparams.height)
        ae_width = int(self.ae.hparams.width)
        if (height, width) != (ae_height, ae_width):
            raise ValueError(
                f"dataset H/W {(height, width)} differs from AE checkpoint {(ae_height, ae_width)}"
            )
        downsample_factor = int(self.ae.net.downsample_factor)
        latent_height = int(height) // downsample_factor
        latent_width = int(width) // downsample_factor
        latent_channels = int(self.ae.hparams.ae_latent_channels)

        mean, std = _load_stats(
            predictor_latent_stats_path,
            latent_channels,
            predictor_latent_mean,
            predictor_latent_std,
        )
        self.register_buffer("latent_mean", mean.view(1, 1, -1, 1, 1))
        self.register_buffer("latent_std", std.view(1, 1, -1, 1, 1))
        # Save numeric stats into hparams so loading a predictor checkpoint does not
        # require the original .npz statistics file.
        self.hparams.predictor_latent_mean = mean.tolist()
        self.hparams.predictor_latent_std = std.tolist()

        self.predictor = LatentSimVPPredictor(
            sequence_length=int(input_length),
            latent_channels=latent_channels,
            latent_height=latent_height,
            latent_width=latent_width,
            hid_s=predictor_hid_s,
            hid_t=predictor_hid_t,
            n_s=predictor_n_s,
            n_t=predictor_n_t,
            incep_ker=predictor_incep_ker,
            groups=predictor_groups,
            residual=predictor_residual,
        )
        self.field_loss = RadarReconstructionLoss(
            radar_vmax=ae_radar_vmax,
            thresholds_dbz=ae_loss_thresholds_dbz,
            weights=ae_loss_weights,
            mse_weight=ae_mse_weight,
            gradient_weight=ae_gradient_weight,
        )

        thresholds = torch.tensor(predictor_metric_thresholds_dbz, dtype=torch.float64)
        self.register_buffer(
            "metric_thresholds", thresholds / float(ae_radar_vmax), persistent=False
        )
        leads = tuple(int(v) for v in predictor_metric_leads)
        if not leads or any(v < 1 or v > target_length for v in leads):
            raise ValueError(f"metric leads must lie in 1..{target_length}, received {leads}")
        self.metric_lead_indices = tuple(v - 1 for v in leads)
        shape = (int(target_length), len(thresholds))
        for name in ("hits", "misses", "false_alarms", "predicted", "observed"):
            self.register_buffer(
                f"val_{name}", torch.zeros(shape, dtype=torch.float64), persistent=False
            )

    def train(self, mode: bool = True):
        super().train(mode)
        # Lightning recursively switches children to train mode; the frozen AE must
        # remain deterministic and must never update any internal state.
        self.ae.eval()
        return self

    def _encode_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 5 or sequence.shape[2] != 1:
            raise ValueError(f"expected BT1HW radar sequence, received {tuple(sequence.shape)}")
        batch, steps, channels, height, width = sequence.shape
        flat = sequence.float().reshape(batch * steps, channels, height, width)
        chunks = []
        with torch.no_grad():
            for chunk in flat.split(int(self.hparams.predictor_encode_chunk_size), dim=0):
                chunks.append(self.ae.encode(chunk))
        latent = torch.cat(chunks, dim=0)
        return latent.reshape(batch, steps, *latent.shape[1:])

    def _normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.latent_mean) / self.latent_std

    def _denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.latent_std + self.latent_mean

    def predict_latent(self, seqs_x: torch.Tensor) -> torch.Tensor:
        z_history = self._normalize(self._encode_sequence(seqs_x))
        return self.predictor(z_history)

    def _decode_sequence(self, normalized_latent: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = normalized_latent.shape
        raw = self._denormalize(normalized_latent)
        flat = raw.reshape(batch * steps, channels, height, width)
        decoded = [
            self.ae.decode(chunk)
            for chunk in flat.split(int(self.hparams.predictor_decode_chunk_size), dim=0)
        ]
        frames = torch.cat(decoded, dim=0)
        return frames.reshape(batch, steps, *frames.shape[1:])

    def forward(self, seqs_x: torch.Tensor) -> torch.Tensor:
        return self._decode_sequence(self.predict_latent(seqs_x))

    def _latent_targets(self, seqs_y: torch.Tensor) -> torch.Tensor:
        return self._normalize(self._encode_sequence(seqs_y))

    def _latent_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(
            prediction,
            target,
            beta=float(self.hparams.predictor_huber_beta),
        )

    def _valid_field(self, frames: torch.Tensor) -> torch.Tensor:
        height = min(int(self.hparams.predictor_metric_height), frames.shape[-2])
        width = min(int(self.hparams.predictor_metric_width), frames.shape[-1])
        return frames[..., :height, :width]

    def _sample_field_loss(
        self, prediction: torch.Tensor, seqs_y: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        count = min(
            int(self.hparams.predictor_field_frames_per_sample), prediction.shape[1]
        )
        indices = torch.randperm(prediction.shape[1], device=prediction.device)[:count]
        selected_latent = prediction.index_select(1, indices)
        selected_target = seqs_y.index_select(1, indices).float()
        decoded = self._decode_sequence(selected_latent)
        decoded = self._valid_field(decoded)
        selected_target = self._valid_field(selected_target)
        return self.field_loss(decoded.flatten(0, 1), selected_target.flatten(0, 1))

    def training_step(self, batch, batch_idx):
        seqs_x, seqs_y = batch
        target_latent = self._latent_targets(seqs_y)
        prediction = self.predict_latent(seqs_x)
        latent_loss = self._latent_loss(prediction, target_latent)
        field_loss = latent_loss.new_zeros(())
        field_parts: dict[str, torch.Tensor] = {}
        if float(self.hparams.predictor_lambda_field) > 0:
            field_loss, field_parts = self._sample_field_loss(prediction, seqs_y)
        total = latent_loss + float(self.hparams.predictor_lambda_field) * field_loss
        metrics = {
            "train_loss": total,
            "train/latent_huber": latent_loss,
            "train/field_loss": field_loss,
        }
        metrics.update({f"train/field_{key}": value for key, value in field_parts.items()})
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=False, batch_size=seqs_x.shape[0])
        self.log("train_loss_step", total, on_step=True, on_epoch=False, prog_bar=True)
        return total

    def on_validation_epoch_start(self) -> None:
        for name in ("hits", "misses", "false_alarms", "predicted", "observed"):
            getattr(self, f"val_{name}").zero_()

    @torch.no_grad()
    def _update_counts(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        metric_h = min(int(self.hparams.predictor_metric_height), prediction.shape[-2])
        metric_w = min(int(self.hparams.predictor_metric_width), prediction.shape[-1])
        prediction = prediction[..., :metric_h, :metric_w]
        target = target[..., :metric_h, :metric_w]
        for lead in range(prediction.shape[1]):
            for threshold_index, threshold in enumerate(self.metric_thresholds):
                pred_event = prediction[:, lead] >= threshold
                true_event = target[:, lead] >= threshold
                hits = torch.logical_and(pred_event, true_event).sum()
                misses = torch.logical_and(~pred_event, true_event).sum()
                false_alarms = torch.logical_and(pred_event, ~true_event).sum()
                self.val_hits[lead, threshold_index] += hits
                self.val_misses[lead, threshold_index] += misses
                self.val_false_alarms[lead, threshold_index] += false_alarms
                self.val_predicted[lead, threshold_index] += pred_event.sum()
                self.val_observed[lead, threshold_index] += true_event.sum()

    def validation_step(self, batch, batch_idx):
        seqs_x, seqs_y = batch
        target_latent = self._latent_targets(seqs_y)
        prediction_latent = self.predict_latent(seqs_x)
        latent_loss = self._latent_loss(prediction_latent, target_latent)
        decoded = self._decode_sequence(prediction_latent)
        decoded_valid = self._valid_field(decoded)
        target_valid = self._valid_field(seqs_y.float())
        field_loss, field_parts = self.field_loss(
            decoded_valid.flatten(0, 1), target_valid.flatten(0, 1)
        )
        total = latent_loss + float(self.hparams.predictor_lambda_field) * field_loss
        clipped = decoded.detach().clamp(0.0, 1.0)
        self._update_counts(clipped, seqs_y.detach())
        self.log_dict(
            {
                "valid_loss_fx": total,
                "val/latent_huber": latent_loss,
                "val/field_loss": field_loss,
                **{f"val/field_{key}": value for key, value in field_parts.items()},
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=seqs_x.shape[0],
        )
        if batch_idx == 0 and int(self.hparams.predictor_log_images):
            self._log_images(clipped, seqs_y.detach())

    @torch.no_grad()
    def _log_images(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if not self.trainer.is_global_zero or self.logger is None:
            return
        experiment = getattr(self.logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "add_images"):
            return
        count = min(int(self.hparams.predictor_num_log_images), prediction.shape[1])
        indices = torch.linspace(0, prediction.shape[1] - 1, count, device=prediction.device).long()
        pred = prediction[0, indices].float().cpu()
        truth = target[0, indices].float().clamp(0.0, 1.0).cpu()
        experiment.add_images("val_predictor/reconstruction", pred, self.global_step)
        experiment.add_images("val_predictor/target", truth, self.global_step)
        experiment.add_images("val_predictor/absolute_error", (pred - truth).abs(), self.global_step)

    @staticmethod
    def _distributed_sum(value: torch.Tensor) -> torch.Tensor:
        value = value.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def on_validation_epoch_end(self) -> None:
        hits = self._distributed_sum(self.val_hits)
        misses = self._distributed_sum(self.val_misses)
        false_alarms = self._distributed_sum(self.val_false_alarms)
        predicted = self._distributed_sum(self.val_predicted)
        observed = self._distributed_sum(self.val_observed)
        eps = torch.finfo(torch.float64).eps
        for threshold_index, threshold in enumerate(self.hparams.predictor_metric_thresholds_dbz):
            label = f"{float(threshold):g}dBZ"
            all_hits = hits[:, threshold_index].sum()
            all_misses = misses[:, threshold_index].sum()
            all_false = false_alarms[:, threshold_index].sum()
            all_predicted = predicted[:, threshold_index].sum()
            all_observed = observed[:, threshold_index].sum()
            aggregate = {
                f"CSI_{label}_val": all_hits / (all_hits + all_misses + all_false + eps),
                f"POD_{label}_val": all_hits / (all_hits + all_misses + eps),
                f"FAR_{label}_val": all_false / (all_hits + all_false + eps),
                f"BIAS_{label}_val": all_predicted / (all_observed + eps),
            }
            self.log_dict(
                {key: value.float() for key, value in aggregate.items()},
                prog_bar=float(threshold) in (35.0, 45.0),
                sync_dist=False,
            )
            for lead_index in self.metric_lead_indices:
                lead_hits = hits[lead_index, threshold_index]
                lead_misses = misses[lead_index, threshold_index]
                lead_false = false_alarms[lead_index, threshold_index]
                lead_predicted = predicted[lead_index, threshold_index]
                lead_observed = observed[lead_index, threshold_index]
                prefix = f"val_lead/{lead_index + 1:02d}_{label}"
                self.log_dict(
                    {
                        f"{prefix}_CSI": (lead_hits / (lead_hits + lead_misses + lead_false + eps)).float(),
                        f"{prefix}_POD": (lead_hits / (lead_hits + lead_misses + eps)).float(),
                        f"{prefix}_FAR": (lead_false / (lead_hits + lead_false + eps)).float(),
                        f"{prefix}_BIAS": (lead_predicted / (lead_observed + eps)).float(),
                    },
                    sync_dist=False,
                )

    def configure_optimizers(self):
        params = [parameter for parameter in self.predictor.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.predictor_weight_decay),
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=int(self.hparams.predictor_lr_patience),
            factor=float(self.hparams.predictor_lr_factor),
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
        parser = parent_parser.add_argument_group("RadarLatentPredictorLit")
        add_parser(parser)
        parser.add_argument("--predictor_ae_checkpoint", type=str, required=True)
        parser.add_argument("--predictor_latent_stats_path", type=str, default="")
        parser.add_argument("--predictor_latent_mean", nargs="+", type=float, default=None)
        parser.add_argument("--predictor_latent_std", nargs="+", type=float, default=None)
        parser.add_argument("--predictor_hid_s", type=int, default=32)
        parser.add_argument("--predictor_hid_t", type=int, default=256)
        parser.add_argument("--predictor_n_s", type=int, default=4)
        parser.add_argument("--predictor_n_t", type=int, default=4)
        parser.add_argument("--predictor_incep_ker", nargs="+", type=int, default=[3, 5, 7, 11])
        parser.add_argument("--predictor_groups", type=int, default=8)
        parser.add_argument("--predictor_residual", choices=["none", "last"], default="none")
        parser.add_argument("--predictor_encode_chunk_size", type=int, default=4)
        parser.add_argument("--predictor_decode_chunk_size", type=int, default=4)
        parser.add_argument("--predictor_huber_beta", type=float, default=1.0)
        parser.add_argument("--predictor_lambda_field", type=float, default=0.0)
        parser.add_argument("--predictor_field_frames_per_sample", type=int, default=2)
        parser.add_argument("--predictor_metric_thresholds_dbz", nargs="+", type=float, default=[25, 35, 45])
        parser.add_argument("--predictor_metric_leads", nargs="+", type=int, default=[1, 5, 10, 15, 20])
        parser.add_argument("--predictor_metric_height", type=int, default=550)
        parser.add_argument("--predictor_metric_width", type=int, default=550)
        parser.add_argument("--predictor_log_images", type=int, default=1)
        parser.add_argument("--predictor_num_log_images", type=int, default=4)
        parser.add_argument("--ae_radar_vmax", type=float, default=70.0)
        parser.add_argument("--ae_loss_thresholds_dbz", nargs="+", type=float, default=[20, 35, 45])
        parser.add_argument("--ae_loss_weights", nargs="+", type=float, default=[1, 2, 4, 8])
        parser.add_argument("--ae_mse_weight", type=float, default=0.2)
        parser.add_argument("--ae_gradient_weight", type=float, default=0.1)
        parser.add_argument("--predictor_weight_decay", type=float, default=1e-4)
        parser.add_argument("--predictor_lr_patience", type=int, default=5)
        parser.add_argument("--predictor_lr_factor", type=float, default=0.5)
        return parent_parser
