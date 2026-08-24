"""雷达单帧 AutoEncoder 个例推理与可视化。

默认从每个样本的全部输入/目标帧中，选择强回波面积最大的若干帧。也可以通过
``--frame_indices`` 手动指定帧号。模型结构参数直接从 Lightning checkpoint 恢复。

示例（在项目根目录运行）：

python inference/infer_radar_autoencoder.py \
  --weight_path /root/private_data/ld_pred/save/radar_ae_f4_c8_xinjiang/weights-epoch=013-valid_loss_fx=0.00298.ckpt \
  --data_file preprocess/xinjiang_cr_6min/data_list_val.csv \
  --radar_dir /root/private_data/ld_pred/data/xinjiang/CR_6min_550x550 \
  --sample_indices 0 10 20 \
  --top_k_frames 4 \
  --device cuda:0 \
  --output_dir inference/radar_ae_f4_examples
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_INFERENCE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _INFERENCE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset.RainDataset import SingleRadarDataset
from models import RadarAutoEncoderLit

torch.set_float32_matmul_precision("medium")


def _resolve_device(requested: str) -> torch.device:
    requested = requested.strip().lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"指定了 --device {requested}，但当前 PyTorch 无可用 CUDA/HIP 设备")
    return torch.device(requested)


def _hparam(hparams: Any, name: str, default: Any = None) -> Any:
    if hasattr(hparams, name):
        return getattr(hparams, name)
    if isinstance(hparams, dict) and name in hparams:
        return hparams[name]
    if default is not None:
        return default
    raise KeyError(f"checkpoint 中缺少超参数 {name!r}")


def _select_frame_indices(
    sequence_tchw: torch.Tensor,
    explicit_indices: list[int] | None,
    top_k: int,
    threshold_dbz: float,
    radar_vmax: float,
) -> list[int]:
    total_frames = int(sequence_tchw.shape[0])
    if explicit_indices:
        invalid = [i for i in explicit_indices if i < 0 or i >= total_frames]
        if invalid:
            raise ValueError(
                f"frame_indices 含非法帧号 {invalid}；当前样本可用范围为 0..{total_frames - 1}"
            )
        return list(dict.fromkeys(int(i) for i in explicit_indices))

    threshold = float(threshold_dbz) / float(radar_vmax)
    frames = sequence_tchw[:, 0]
    high_area = (frames >= threshold).flatten(1).sum(dim=1)
    peak = frames.flatten(1).amax(dim=1)
    # 先按强回波面积、再按峰值排序，优先检查最难重建的对流核心。
    ranked = sorted(
        range(total_frames),
        key=lambda i: (int(high_area[i]), float(peak[i])),
        reverse=True,
    )
    return ranked[: min(int(top_k), total_frames)]


def _event_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold_dbz: float,
    radar_vmax: float,
) -> dict[str, float | None]:
    threshold = float(threshold_dbz) / float(radar_vmax)
    pred_event = prediction >= threshold
    true_event = target >= threshold
    hits = int(torch.logical_and(pred_event, true_event).sum().item())
    misses = int(torch.logical_and(~pred_event, true_event).sum().item())
    false_alarms = int(torch.logical_and(pred_event, ~true_event).sum().item())
    predicted = hits + false_alarms
    observed = hits + misses
    union = hits + misses + false_alarms

    def ratio(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    return {
        "csi": ratio(hits, union),
        "pod": ratio(hits, observed),
        "far": ratio(false_alarms, predicted),
        "bias": ratio(predicted, observed),
    }


def _frame_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    thresholds_dbz: list[float],
    radar_vmax: float,
) -> dict[str, Any]:
    error_dbz = (prediction - target) * float(radar_vmax)
    result: dict[str, Any] = {
        "mae_dbz": float(error_dbz.abs().mean().item()),
        "rmse_dbz": float(torch.sqrt(torch.square(error_dbz).mean()).item()),
        "target_max_dbz": float(target.max().item() * radar_vmax),
        "prediction_max_dbz": float(prediction.max().item() * radar_vmax),
    }
    result["peak_error_dbz"] = result["prediction_max_dbz"] - result["target_max_dbz"]
    result["thresholds"] = {
        f"{threshold:g}": _event_metrics(prediction, target, threshold, radar_vmax)
        for threshold in thresholds_dbz
    }
    return result


def _plot_frame(
    target: torch.Tensor,
    prediction: torch.Tensor,
    output_path: Path,
    title: str,
    radar_vmax: float,
    crop_height: int | None,
    crop_width: int | None,
) -> None:
    target_dbz = target.detach().float().cpu().numpy() * radar_vmax
    prediction_dbz = prediction.detach().float().cpu().numpy() * radar_vmax
    if crop_height is not None:
        target_dbz = target_dbz[:crop_height, :]
        prediction_dbz = prediction_dbz[:crop_height, :]
    if crop_width is not None:
        target_dbz = target_dbz[:, :crop_width]
        prediction_dbz = prediction_dbz[:, :crop_width]

    signed_error = prediction_dbz - target_dbz
    absolute_error = np.abs(signed_error)
    signed_limit = max(5.0, min(30.0, float(np.percentile(absolute_error, 99.5))))
    absolute_limit = max(5.0, min(30.0, float(np.percentile(absolute_error, 99.5))))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), constrained_layout=True)
    panels = (
        (target_dbz, "Target dBZ", "turbo", 0.0, radar_vmax),
        (prediction_dbz, "Reconstruction dBZ", "turbo", 0.0, radar_vmax),
        (signed_error, "Signed error dBZ", "coolwarm", -signed_limit, signed_limit),
        (absolute_error, "Absolute error dBZ", "magma", 0.0, absolute_limit),
    )
    for axis, (image, label, cmap, vmin, vmax) in zip(axes, panels):
        shown = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(label)
        axis.axis("off")
        fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.025)
    fig.suptitle(title, fontsize=11)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {
            key: value
            for key, value in record.items()
            if key != "thresholds"
        }
        for threshold, metrics in record["thresholds"].items():
            for metric_name, value in metrics.items():
                row[f"{metric_name}_{threshold}dbz"] = value
        rows.append(row)
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="雷达单帧 AutoEncoder 个例推理")
    parser.add_argument("--weight_path", required=True, help="Lightning .ckpt 路径")
    parser.add_argument("--data_file", required=True, help="验证集或测试集 CSV")
    parser.add_argument("--radar_dir", required=True, help="雷达数据根目录")
    parser.add_argument("--sample_indices", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--frame_indices",
        type=int,
        nargs="+",
        default=None,
        help="在 input+target 拼接序列中的帧号；省略则自动选强回波面积最大的帧",
    )
    parser.add_argument("--top_k_frames", type=int, default=4)
    parser.add_argument("--selection_threshold_dbz", type=float, default=35.0)
    parser.add_argument("--metric_thresholds_dbz", type=float, nargs="+", default=[25, 35, 45])
    parser.add_argument("--packed_sequence_file", type=int, choices=(0, 1), default=0)
    parser.add_argument("--packed_event_npy", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda:0 等")
    parser.add_argument("--output_dir", default="inference/radar_ae_examples")
    parser.add_argument("--crop_height", type=int, default=None)
    parser.add_argument("--crop_width", type=int, default=None)
    args = parser.parse_args()

    if args.packed_sequence_file and args.packed_event_npy:
        parser.error("--packed_sequence_file 与 --packed_event_npy 不能同时为 1")
    if args.top_k_frames <= 0:
        parser.error("--top_k_frames 必须为正数")

    weight_path = Path(args.weight_path).expanduser().resolve()
    data_file = Path(args.data_file).expanduser().resolve()
    radar_dir = Path(args.radar_dir).expanduser().resolve()
    for path, label in (
        (weight_path, "checkpoint"),
        (data_file, "data_file"),
        (radar_dir, "radar_dir"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} 不存在: {path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    model = RadarAutoEncoderLit.load_from_checkpoint(
        str(weight_path),
        map_location="cpu",
    )
    model.eval().to(device)
    hparams = model.hparams
    input_length = int(_hparam(hparams, "input_length", 20))
    target_length = int(_hparam(hparams, "target_length", 20))
    height = int(_hparam(hparams, "height"))
    width = int(_hparam(hparams, "width"))
    radar_vmax = float(_hparam(hparams, "ae_radar_vmax", 70.0))

    dataset = SingleRadarDataset(
        str(data_file),
        str(radar_dir),
        input_length=input_length,
        target_length=target_length,
        height=height,
        width=width,
        packed_sequence_file=bool(args.packed_sequence_file),
        packed_event_npy=bool(args.packed_event_npy),
    )

    invalid_samples = [i for i in args.sample_indices if i < 0 or i >= len(dataset)]
    if invalid_samples:
        raise IndexError(
            f"sample_indices 含非法索引 {invalid_samples}；数据集长度为 {len(dataset)}"
        )

    records: list[dict[str, Any]] = []
    for sample_index in args.sample_indices:
        seq_x, seq_y = dataset[int(sample_index)]
        sequence = torch.from_numpy(np.concatenate((seq_x, seq_y), axis=0)).float()
        selected_indices = _select_frame_indices(
            sequence,
            args.frame_indices,
            args.top_k_frames,
            args.selection_threshold_dbz,
            radar_vmax,
        )
        frames = sequence[selected_indices].to(device)
        with torch.inference_mode():
            latents = model.encode(frames)
            reconstructions = model.decode(latents).clamp(0.0, 1.0)

        for batch_index, frame_index in enumerate(selected_indices):
            target = frames[batch_index, 0]
            prediction = reconstructions[batch_index, 0]
            metrics = _frame_metrics(
                prediction,
                target,
                args.metric_thresholds_dbz,
                radar_vmax,
            )
            source = "input" if frame_index < input_length else "target"
            source_index = frame_index if source == "input" else frame_index - input_length
            record = {
                "sample_index": int(sample_index),
                "frame_index": int(frame_index),
                "source": source,
                "source_frame_index": int(source_index),
                **metrics,
            }
            records.append(record)

            stem = f"sample_{sample_index:05d}_frame_{frame_index:02d}"
            title = (
                f"sample={sample_index}, frame={frame_index} ({source}[{source_index}]) | "
                f"MAE={metrics['mae_dbz']:.3f} dBZ, RMSE={metrics['rmse_dbz']:.3f} dBZ | "
                f"peak {metrics['target_max_dbz']:.1f} -> "
                f"{metrics['prediction_max_dbz']:.1f} dBZ"
            )
            _plot_frame(
                target,
                prediction,
                output_dir / f"{stem}.png",
                title,
                radar_vmax,
                args.crop_height,
                args.crop_width,
            )
            np.savez_compressed(
                output_dir / f"{stem}.npz",
                target=target.detach().cpu().numpy(),
                reconstruction=prediction.detach().cpu().numpy(),
                latent=latents[batch_index].detach().cpu().numpy(),
            )

    summary = {
        "weight_path": str(weight_path),
        "data_file": str(data_file),
        "radar_dir": str(radar_dir),
        "device": str(device),
        "downsample_factor": int(model.net.downsample_factor),
        "latent_channels": int(_hparam(hparams, "ae_latent_channels", 8)),
        "radar_vmax": radar_vmax,
        "records": records,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    _write_csv(records, output_dir / "metrics.csv")

    print(
        f"完成：{len(records)} 个单帧个例，模型 f={summary['downsample_factor']}，"
        f"结果保存在 {output_dir}"
    )
    print(json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
