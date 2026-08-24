# Earthformer（EarthformerLit / CuboidTransformer）在 Moving MNIST 固定测试集（mnist_test_seq.npy）上的离线推理与可视化。
# 流程与 inference/infer_simvp_mmnist.py、infer_convlstm_mmnist.py 一致：读 npy、与 MovingMNIST_Phys 固定集预处理对齐，
# Lightning ckpt 加载后 model(seqs_x) 推理（内部 B,T,C,H,W -> B,T,H,W,C），汇总 MSE/MAE（含逐帧），曲线与样本图写入 output_dir。
#
# 用法（在 transfer 目录下，PyTorch 任务建议 conda activate torch）:
#   python inference/infer_earthformer_mmnist.py --weight_path save/earthformer_mmnist/weights-xxx.ckpt \
#     --root_dir data/moving_mnist \
#     --height 64 --width 64 --input_length 10 --target_length 10 \
#     --input_class 0 --predict_class 0 --predict_class_vmax 1 \
#     --batch_size 8 --accelerator gpu --gpus 0 --max_samples 10
# 昇腾: --accelerator npu --gpus 1
# 可选: --earthformer_oc_file 自定义 yaml（须与训练时一致；空则从 ckpt/默认 models/earthformer_default_mmnist.yaml）；
#       --npy_path；--output_dir（默认 transfer/inference/earthformer_mmnist_infer_out）；
#       --max_samples；--save_pred_npy。

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

_INFERENCE_DIR = Path(__file__).resolve().parent
_TRANSFER_ROOT = _INFERENCE_DIR.parent
if str(_TRANSFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRANSFER_ROOT))

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.earthformer_lit import EarthformerLit
from train import _trainer_accelerator_devices_strategy

torch.set_float32_matmul_precision("medium")

_DEFAULT_OUTPUT_DIR = str(_INFERENCE_DIR / "earthformer_mmnist_infer_out")


def load_fixed_npy(path: str) -> np.ndarray:
    """与 MovingMnist.load_fixed_set 一致：加载后增加 channel 维。"""
    dataset = np.load(path)
    return dataset[..., np.newaxis]


def raw_seq_to_tensors(
    images: np.ndarray,
    n_frames_input: int,
    n_frames_output: int,
    image_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """与 MovingMNIST_Phys.__getitem__ 中固定集分支的 reshape /255 一致。"""
    length = n_frames_input + n_frames_output
    assert images.shape[0] == length, f"期望时间维 {length}, 得到 {images.shape[0]}"
    r = 1
    w = int(image_size / r)
    x = images.reshape((length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape((length, r * r, w, w))
    inp = x[:n_frames_input]
    out = x[n_frames_input:length]
    inp_t = torch.from_numpy(inp / 255.0).contiguous().float()
    out_t = torch.from_numpy(out / 255.0).contiguous().float()
    return inp_t, out_t


def batch_from_indices(
    dataset: np.ndarray,
    indices: List[int],
    n_frames_input: int,
    n_frames_output: int,
    image_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """dataset 形状 (T_total, N, H, W, C)。"""
    xs, ys = [], []
    for idx in indices:
        images = dataset[:, idx, ...]
        inp, out = raw_seq_to_tensors(images, n_frames_input, n_frames_output, image_size)
        xs.append(inp)
        ys.append(out)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def resolve_torch_device(accelerator: str, devices) -> torch.device:
    acc = accelerator.lower()
    if acc == "cpu":
        return torch.device("cpu")
    if acc == "gpu":
        if isinstance(devices, list):
            idx = int(devices[0])
        else:
            s = str(devices).strip()
            if s == "-1":
                idx = 0
            elif "," in s:
                idx = int(s.split(",")[0].strip())
            else:
                idx = int(s)
        return torch.device(f"cuda:{idx}")
    if acc == "npu":
        return torch.device("npu:0")
    return torch.device("cpu")


def plot_per_frame_curve(
    values: np.ndarray,
    ylabel: str,
    title: str,
    out_path: str,
) -> None:
    t = np.arange(1, len(values) + 1)
    plt.figure(figsize=(6, 3.5))
    plt.plot(t, values, marker="o", markersize=4)
    plt.xlabel("预测帧索引 (1-based)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_sample_comparison(
    seq_x: torch.Tensor,
    seq_y: torch.Tensor,
    pred: torch.Tensor,
    out_path: str,
) -> None:
    """seq_x: (T_in,C,H,W), seq_y/pred: (T_out,C,H,W)，均在 CPU。"""
    t_out = seq_y.shape[0]
    fig = plt.figure(figsize=(5, 2.2 * (t_out + 1)))
    gs = gridspec.GridSpec(t_out + 1, 2, figure=fig)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(seq_x[-1, 0].numpy(), cmap="gray", vmin=0, vmax=1)
    ax0.set_title("条件输入：末帧")
    ax0.axis("off")
    for t in range(t_out):
        ax_g = fig.add_subplot(gs[t + 1, 0])
        ax_g.imshow(seq_y[t, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax_g.set_title(f"帧 {t + 1} GT")
        ax_g.axis("off")
        ax_p = fig.add_subplot(gs[t + 1, 1])
        ax_p.imshow(pred[t, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax_p.set_title(f"帧 {t + 1} Pred")
        ax_p.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def main() -> None:
    parser = ArgumentParser(description="Earthformer Moving MNIST 固定测试集推理")
    parser.add_argument("--model_name", type=str, default="earthformer")
    parser.add_argument("--weight_path", type=str, required=True, help="Lightning ckpt 路径")
    parser.add_argument("--root_dir", type=str, default=None, help="含 mnist_test_seq.npy 的目录")
    parser.add_argument(
        "--npy_path",
        type=str,
        default=None,
        help="直接指定 mnist_test_seq.npy；若设置则忽略 root_dir 拼接",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"指标与图像输出目录（默认: {_DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None, help="最多评估的序列条数，默认全量")
    parser.add_argument("--num_plot_samples", type=int, default=5, help="保存 sample_XX.png 的条数")
    parser.add_argument("--save_pred_npy", action="store_true", help="将前 num_plot_samples 条 pred/gt 存为 npy")
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=("gpu", "npu", "cpu"),
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="设备数量或列表（如 1 或 0,1）；留空时 GPU 沿用 --gpus",
    )
    parser.add_argument("--gpus", type=str, default="0")
    parser = EarthformerLit.add_model_specific_args(parser)
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)

    if args.npy_path:
        npy_file = args.npy_path
    else:
        if not args.root_dir:
            raise SystemExit("请指定 --npy_path 或 --root_dir")
        npy_file = os.path.join(args.root_dir, "mnist_test_seq.npy")
    if not os.path.isfile(npy_file):
        raise SystemExit(f"找不到测试 npy: {npy_file}")

    os.makedirs(output_dir, exist_ok=True)

    acc, dev, _ = _trainer_accelerator_devices_strategy(args.accelerator, args.devices, args.gpus)
    device = resolve_torch_device(acc, dev)

    dataset = load_fixed_npy(npy_file)
    n_seq = int(dataset.shape[1])
    t_total = args.input_length + args.target_length
    if int(dataset.shape[0]) != t_total:
        raise SystemExit(
            f"npy 时间维 {dataset.shape[0]} 与 input_length+target_length={t_total} 不一致"
        )

    n_eval = n_seq if args.max_samples is None else min(n_seq, int(args.max_samples))

    dict_args = vars(args).copy()
    _infer_only = frozenset({
        "weight_path",
        "root_dir",
        "npy_path",
        "output_dir",
        "max_samples",
        "num_plot_samples",
        "save_pred_npy",
        "accelerator",
        "devices",
        "gpus",
        "model_name",
    })
    model_kwargs = {k: v for k, v in dict_args.items() if k not in _infer_only}
    model = EarthformerLit.load_from_checkpoint(
        args.weight_path,
        strict=False,
        map_location="cpu",
        **model_kwargs,
    )
    model.to(device)
    model.eval()

    t_out = args.target_length
    c = len(args.predict_class)
    h, w = args.height, args.width

    sum_sq_all = 0.0
    sum_abs_all = 0.0
    n_elem_total = 0
    sum_sq_frame = np.zeros(t_out, dtype=np.float64)
    sum_abs_frame = np.zeros(t_out, dtype=np.float64)

    plot_store: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    with torch.no_grad():
        start = 0
        while start < n_eval:
            end = min(start + args.batch_size, n_eval)
            idxs = list(range(start, end))
            seqs_x, seqs_y = batch_from_indices(
                dataset,
                idxs,
                args.input_length,
                args.target_length,
                image_size=h,
            )
            seqs_x = seqs_x.to(device)
            seqs_y = seqs_y.to(device)
            pred = model(seqs_x)
            pred = torch.clip(pred, 0, 1)

            diff = pred - seqs_y
            b = pred.shape[0]
            sum_sq_all += float(torch.sum(diff * diff).item())
            sum_abs_all += float(torch.sum(torch.abs(diff)).item())
            n_elem_total += b * t_out * c * h * w

            for t in range(t_out):
                frame_diff = pred[:, t, ...] - seqs_y[:, t, ...]
                sum_sq_frame[t] += float(torch.sum(frame_diff * frame_diff).item())
                sum_abs_frame[t] += float(torch.sum(torch.abs(frame_diff)).item())

            for i, global_i in enumerate(idxs):
                if global_i < args.num_plot_samples:
                    plot_store[global_i] = (
                        seqs_x[i].detach().cpu(),
                        seqs_y[i].detach().cpu(),
                        pred[i].detach().cpu(),
                    )

            if args.save_pred_npy:
                for i, global_i in enumerate(idxs):
                    if global_i < args.num_plot_samples:
                        base = os.path.join(output_dir, f"sample_{global_i:02d}")
                        np.save(f"{base}_pred.npy", pred[i].detach().cpu().numpy())
                        np.save(f"{base}_gt.npy", seqs_y[i].detach().cpu().numpy())

            start = end

    mse_global = sum_sq_all / max(n_elem_total, 1)
    mae_global = sum_abs_all / max(n_elem_total, 1)
    mse_per_frame = sum_sq_frame / max(n_eval, 1)
    mae_per_frame = sum_abs_frame / max(n_eval, 1)

    metrics = {
        "npy_path": npy_file,
        "weight_path": args.weight_path,
        "output_dir": output_dir,
        "n_eval_sequences": n_eval,
        "mse_global": float(mse_global),
        "mae_global": float(mae_global),
        "mse_per_frame": mse_per_frame.tolist(),
        "mae_per_frame": mae_per_frame.tolist(),
    }

    txt_path = os.path.join(output_dir, "metrics.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics, indent=2, ensure_ascii=False))
    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    plot_per_frame_curve(
        mae_per_frame,
        "MAE（空间上 sum|err|/B 再对全体 N 条序列平均）",
        "逐帧 MAE（与 EarthformerLit.validation_step 中 clip 后 MSE 同尺度对比用）",
        os.path.join(output_dir, "per_frame_mae.png"),
    )
    plot_per_frame_curve(
        mse_per_frame,
        "MSE（空间上 sum err^2/B 再对全体 N 条序列平均）",
        "逐帧 MSE（与 EarthformerLit.validation_step 中 mse 定义一致）",
        os.path.join(output_dir, "per_frame_mse.png"),
    )

    for i in range(min(args.num_plot_samples, n_eval)):
        if i not in plot_store:
            continue
        sx, sy, pr = plot_store[i]
        plot_sample_comparison(
            sx,
            sy,
            pr,
            os.path.join(output_dir, f"sample_{i:02d}.png"),
        )


if __name__ == "__main__":
    main()
