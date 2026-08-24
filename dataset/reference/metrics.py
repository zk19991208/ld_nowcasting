"""Radar nowcast evaluation metrics: CSI, POD, FAR.

All metrics are pixel-level binary classification metrics computed by
thresholding normalized predictions / targets back to original dBZ space.
Supports multiple thresholds and per-lead-time breakdown.
"""

from typing import Dict, List, Tuple

import torch
from torchmetrics import Metric


class RadarMetrics(Metric):
    """Accumulates TP / FP / FN across batches then computes CSI, POD, FAR.

    Operates on **normalized** [0, 1] tensors.  Thresholds are given in
    original dBZ values and converted to [0, 1] internally.

    Args:
        thresholds_dbz: dBZ thresholds for binary conversion.
        vmax: max value used during normalization (default 70 for dBZ).
        per_leadtime: if True, also keep per-lead-time accumulators.
            Requires that ``update`` receives ``(T_out, ...)`` shaped data
            or that the caller reshapes accordingly.
        n_output_frames: number of output time steps (needed when per_leadtime=True).
        n_vars: number of variables (channels interleaved per time step).
    """

    full_state_update = False

    def __init__(
        self,
        thresholds_dbz: Tuple[float, ...] = (20, 35, 40),
        vmax: float = 70.0,
        per_leadtime: bool = False,
        n_output_frames: int = 20,
        n_vars: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_thresholds = [t / vmax for t in thresholds_dbz]
        self.threshold_names = [f"{int(t)}dBZ" for t in thresholds_dbz]
        self.per_leadtime = per_leadtime
        self.n_output_frames = n_output_frames
        self.n_vars = n_vars

        for name in self.threshold_names:
            self.add_state(f"tp_{name}", default=torch.zeros(1), dist_reduce_fx="sum")
            self.add_state(f"fp_{name}", default=torch.zeros(1), dist_reduce_fx="sum")
            self.add_state(f"fn_{name}", default=torch.zeros(1), dist_reduce_fx="sum")

            if self.per_leadtime:
                self.add_state(
                    f"tp_lt_{name}",
                    default=torch.zeros(n_output_frames),
                    dist_reduce_fx="sum",
                )
                self.add_state(
                    f"fp_lt_{name}",
                    default=torch.zeros(n_output_frames),
                    dist_reduce_fx="sum",
                )
                self.add_state(
                    f"fn_lt_{name}",
                    default=torch.zeros(n_output_frames),
                    dist_reduce_fx="sum",
                )

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update accumulators.

        Args:
            pred:   (B, T_out * C, H, W) or (B, T_out, H, W) when C=1.
            target: same shape as pred.
        """
        B = pred.shape[0]
        C = self.n_vars
        T = self.n_output_frames

        if self.per_leadtime:
            pred_4d = pred.view(B, T, C, pred.shape[-2], pred.shape[-1])
            tgt_4d = target.view(B, T, C, target.shape[-2], target.shape[-1])
            # Only evaluate the first variable (dBZ) for per-lead-time metrics
            pred_lt = pred_4d[:, :, 0]  # (B, T, H, W)
            tgt_lt = tgt_4d[:, :, 0]

        for thresh, name in zip(self.norm_thresholds, self.threshold_names):
            p_bin = pred >= thresh
            t_bin = target >= thresh

            tp = (p_bin & t_bin).sum()
            fp = (p_bin & ~t_bin).sum()
            fn = (~p_bin & t_bin).sum()

            getattr(self, f"tp_{name}").add_(tp)
            getattr(self, f"fp_{name}").add_(fp)
            getattr(self, f"fn_{name}").add_(fn)

            if self.per_leadtime:
                p_lt_bin = pred_lt >= thresh
                t_lt_bin = tgt_lt >= thresh
                for t_idx in range(T):
                    tp_t = (p_lt_bin[:, t_idx] & t_lt_bin[:, t_idx]).sum()
                    fp_t = (p_lt_bin[:, t_idx] & ~t_lt_bin[:, t_idx]).sum()
                    fn_t = (~p_lt_bin[:, t_idx] & t_lt_bin[:, t_idx]).sum()
                    getattr(self, f"tp_lt_{name}")[t_idx] += tp_t
                    getattr(self, f"fp_lt_{name}")[t_idx] += fp_t
                    getattr(self, f"fn_lt_{name}")[t_idx] += fn_t

    def compute(self) -> Dict[str, torch.Tensor]:
        results = {}
        eps = 1e-8
        for name in self.threshold_names:
            tp = getattr(self, f"tp_{name}").float()
            fp = getattr(self, f"fp_{name}").float()
            fn = getattr(self, f"fn_{name}").float()

            results[f"CSI_{name}"] = tp / (tp + fp + fn + eps)
            results[f"POD_{name}"] = tp / (tp + fn + eps)
            results[f"FAR_{name}"] = fp / (tp + fp + eps)

            if self.per_leadtime:
                tp_lt = getattr(self, f"tp_lt_{name}").float()
                fp_lt = getattr(self, f"fp_lt_{name}").float()
                fn_lt = getattr(self, f"fn_lt_{name}").float()

                csi_lt = tp_lt / (tp_lt + fp_lt + fn_lt + eps)
                pod_lt = tp_lt / (tp_lt + fn_lt + eps)
                far_lt = fp_lt / (tp_lt + fp_lt + eps)

                results[f"CSI_lt_{name}"] = csi_lt
                results[f"POD_lt_{name}"] = pod_lt
                results[f"FAR_lt_{name}"] = far_lt

        return results
