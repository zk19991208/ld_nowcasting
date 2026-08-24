"""Lightning callbacks for radar nowcast training visualization."""

import io
import logging
import os
from typing import List, Optional, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from PIL import Image
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)

# Standard radar reflectivity colour map boundaries (dBZ, normalised to [0,1] with vmax=70)
_DBZ_BOUNDS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
_DBZ_COLORS = [
    "#FFFFFF", "#A6F0F0", "#00C8C8", "#00A000", "#00E000",
    "#FFFF00", "#E0C800", "#FF9000", "#FF0000", "#C80000",
    "#800080", "#C000C0", "#FF00FF", "#800040",
]
_NORM_BOUNDS = [b / 70.0 for b in _DBZ_BOUNDS]
_RADAR_CMAP = mcolors.ListedColormap(_DBZ_COLORS)
_RADAR_NORM = mcolors.BoundaryNorm(_NORM_BOUNDS, _RADAR_CMAP.N)


def _fig_to_numpy(fig) -> np.ndarray:
    """Render a matplotlib figure to an (H, W, 3) uint8 numpy array."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"))
    buf.close()
    return img


def _build_comparison_figure(
    pred_np: np.ndarray,
    tgt_np: np.ndarray,
    n_output: int,
    n_vars: int,
    display_lead_minutes: List[int],
    frame_interval_min: int,
    title: str,
):
    """Build a 2-row (Pred / Truth) comparison figure and return the Figure object."""
    pred_4d = pred_np.reshape(n_output, n_vars, pred_np.shape[-2], pred_np.shape[-1])
    tgt_4d = tgt_np.reshape(n_output, n_vars, tgt_np.shape[-2], tgt_np.shape[-1])
    pred_frames = pred_4d[:, 0]  # (T_out, H, W) – first variable (dBZ)
    tgt_frames = tgt_4d[:, 0]

    frame_indices = []
    for lead in display_lead_minutes:
        idx = lead // frame_interval_min - 1
        if 0 <= idx < n_output:
            frame_indices.append(idx)

    n_cols = len(frame_indices)
    cell_w = max(1.3, min(2.6, 36.0 / n_cols))
    fig_w = cell_w * n_cols + 1.0
    fig_h = cell_w * 2 + 0.6
    title_fs = 11 if n_cols > 10 else 13
    label_fs = 8 if n_cols > 10 else 10

    fig, axes = plt.subplots(
        2, n_cols + 1,
        figsize=(fig_w, fig_h), dpi=100,
        gridspec_kw={"width_ratios": [1] * n_cols + [0.06]},
    )
    if n_cols == 1:
        axes = axes.reshape(2, -1)

    for col, fidx in enumerate(frame_indices):
        lead_min = (fidx + 1) * frame_interval_min

        axes[0, col].imshow(
            pred_frames[fidx], cmap=_RADAR_CMAP, norm=_RADAR_NORM,
            interpolation="nearest",
        )
        axes[0, col].set_title(f"+{lead_min}′", fontsize=label_fs)
        axes[0, col].axis("off")

        axes[1, col].imshow(
            tgt_frames[fidx], cmap=_RADAR_CMAP, norm=_RADAR_NORM,
            interpolation="nearest",
        )
        axes[1, col].set_title(f"+{lead_min}′", fontsize=label_fs)
        axes[1, col].axis("off")

    axes[0, 0].text(
        -0.15, 0.5, "Pred", transform=axes[0, 0].transAxes,
        fontsize=label_fs + 2, fontweight="bold", va="center", rotation=90,
    )
    axes[1, 0].text(
        -0.15, 0.5, "Truth", transform=axes[1, 0].transAxes,
        fontsize=label_fs + 2, fontweight="bold", va="center", rotation=90,
    )

    sm = plt.cm.ScalarMappable(norm=_RADAR_NORM, cmap=_RADAR_CMAP)
    axes[0, -1].axis("off")
    cbar = fig.colorbar(sm, ax=axes[1, -1], fraction=1.0, pad=0.0)
    axes[1, -1].axis("off")
    cbar.set_ticks(_NORM_BOUNDS[::2])
    cbar.set_ticklabels([f"{int(b)}" for b in _DBZ_BOUNDS[::2]])
    cbar.set_label("dBZ", fontsize=label_fs - 1)

    fig.suptitle(title, fontsize=title_fs)
    fig.tight_layout(rect=[0.02, 0.0, 1.0, 0.96])

    return fig


def _log_figure_to_loggers(trainer, tag: str, fig, global_step: int):
    """Send a matplotlib figure to all attached loggers (TensorBoard / WandB)."""
    if trainer.logger is None:
        return

    loggers = trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]

    for lg in loggers:
        cls_name = type(lg).__name__
        if cls_name == "TensorBoardLogger":
            lg.experiment.add_figure(tag, fig, global_step=global_step)
        elif cls_name == "WandbLogger":
            import wandb
            img_np = _fig_to_numpy(fig)
            lg.experiment.log({tag: wandb.Image(img_np)}, step=global_step)


class NowcastVisualizationCallback(Callback):
    """Log prediction-vs-truth comparison images during training.

    Layout: 2 rows x N columns.
      Row 0 = model prediction
      Row 1 = ground truth label
      Each column = one lead-time step.

    Args:
        display_lead_minutes: which lead times (in minutes) to display.
        frame_interval_min: minutes between consecutive output frames (6).
        every_n_epochs: produce val figures every N epochs.
        val_every_n_steps: produce val figures every N steps within an epoch.
            0 means only at the end of the epoch.
        train_every_n_steps: produce training figures every N steps within
            an epoch.  0 disables training visualisation.
        save_dir: if set, also save PNG files to this directory.
    """

    def __init__(
        self,
        display_lead_minutes: Optional[List[int]] = None,
        frame_interval_min: int = 6,
        every_n_epochs: int = 1,
        val_every_n_steps: int = 0,
        train_every_n_steps: int = 0,
        save_dir: Optional[str] = None,
    ):
        super().__init__()
        if display_lead_minutes is None:
            display_lead_minutes = [6, 12, 30, 60, 90, 120]
        self.display_lead_minutes = display_lead_minutes
        self.frame_interval_min = frame_interval_min
        self.every_n_epochs = every_n_epochs
        self.val_every_n_steps = val_every_n_steps
        self.train_every_n_steps = train_every_n_steps
        self.save_dir = save_dir

    def _is_active_epoch(self, trainer) -> bool:
        return trainer.current_epoch % self.every_n_epochs == 0

    def _generate_and_log(self, trainer, pl_module, batch, stage: str, batch_idx: int):
        if trainer.global_rank != 0:
            return

        inp, tgt = batch[0][:1], batch[1][:1]
        with torch.no_grad():
            pred = pl_module(inp.to(pl_module.device))

        pred_np = pred[0].float().cpu().numpy()
        tgt_np = tgt[0].float().cpu().numpy()

        n_vars = len(pl_module.hparams["data"]["variables"])
        n_output = pl_module.hparams["data"]["n_output_frames"]

        title = f"Epoch {trainer.current_epoch}  {stage} step {batch_idx}"
        fig = _build_comparison_figure(
            pred_np, tgt_np, n_output, n_vars,
            self.display_lead_minutes, self.frame_interval_min, title,
        )

        tag = f"{stage}/pred_vs_truth"
        _log_figure_to_loggers(trainer, tag, fig, global_step=trainer.global_step)

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            fname = f"ep{trainer.current_epoch:03d}_{stage}_step{batch_idx:05d}.png"
            fig.savefig(os.path.join(self.save_dir, fname), bbox_inches="tight")

        plt.close(fig)

    # ── Training steps ───────────────────────────────────────────────
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.train_every_n_steps <= 0:
            return
        if not self._is_active_epoch(trainer):
            return
        if batch_idx % self.train_every_n_steps == 0:
            self._generate_and_log(trainer, pl_module, batch, "train", batch_idx)

    # ── Validation steps ─────────────────────────────────────────────
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self._is_active_epoch(trainer):
            return
        if self.val_every_n_steps > 0:
            if batch_idx % self.val_every_n_steps == 0:
                self._generate_and_log(trainer, pl_module, batch, "val", batch_idx)
        elif batch_idx == 0:
            self._generate_and_log(trainer, pl_module, batch, "val", batch_idx)
