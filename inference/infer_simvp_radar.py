# SimVP 雷达外推离线推理：与 train.py 中 SingleRadarDataModule 验证集相同的数据读取方式，
# 从 val CSV 指定若干样本索引，加载 ckpt 推理，输出逐样本 MSE/MAE 及「输入 / 真值 / 预报」栅格拼图 PNG。
#
# 主要用法（在 transfer 目录下；GPU 用 conda activate torch，昇腾 NPU 用已装 torch_npu 的环境）:
# python inference/infer_simvp_radar.py --weight_path .../weights-epoch=013-valid_loss_fx=0.023.ckpt \
#   --val_file preprocess/huadong_cr_6min/data_list_val.csv \
#   --radar_dir /home/user/data/huadong/CR_6min_550x550 \
#   --packed_event_npy 0 --val_indices 0 1 --output_dir inference/radar_infer_out \
#   --accelerator npu --gpus 0
# 华为昇腾 NPU（Linux 上先 source CANN set_env.sh，与 ascend_run/README_ASCEND.md 一致）:
#   python inference/infer_simvp_radar.py --accelerator npu --gpus 0 ...（其余同训练时的模型参数）
#   多卡时选某张 NPU：--accelerator npu --devices 1 或 --gpus 1（0 为第一张卡）。
# 事件帧 .npy 模式（与 ascend_run/configs/simvp_huadong_cr_550.yaml 一致）:
#   ... --packed_event_npy 1 --radar_dir .../val --val_file .../data_list_val_events.csv
# 不指定 --val_indices 时默认取验证集第 0、1 条。

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset

try:
    import torch_npu  # noqa: F401 — 昇腾上必须导入以注册 torch.device("npu")
except ImportError:
    torch_npu = None

_INFERENCE_DIR = Path(__file__).resolve().parent
_TRANSFER_ROOT = _INFERENCE_DIR.parent
if str(_TRANSFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRANSFER_ROOT))

from dataset.RainDataset import SingleRadarDataset
from models import SimVP_Lit

torch.set_float32_matmul_precision("medium")

_DEFAULT_OUT = str(_INFERENCE_DIR / "radar_infer_out")


def _parse_device_index(devices, gpus: str) -> int:
    """与 Lightning 习惯一致：--devices 优先，否则解析 --gpus（如 0 或 0,1 取第一张）。"""
    if devices is not None:
        return int(devices)
    s = str(gpus).strip()
    if s in ("", "-1", "None", "none"):
        return 0
    if "," in s:
        return int(s.split(",")[0].strip())
    return int(s)


def _resolve_device(accelerator: str, devices, gpus: str) -> torch.device:
    acc = (accelerator or "gpu").lower()
    if acc == "cpu":
        return torch.device("cpu")
    idx = _parse_device_index(devices, gpus)
    if acc == "npu":
        if torch_npu is None:
            raise RuntimeError(
                "当前选择了 --accelerator npu，但未安装/未能 import torch_npu。"
                "请在昇腾环境安装与 CANN 匹配的 torch_npu，并先执行 CANN 的 set_env.sh。"
            )
        return torch.device(f"npu:{idx}")
    return torch.device(f"cuda:{idx}")


def plot_sample_triple(
    seq_x: torch.Tensor,
    seq_y: torch.Tensor,
    pred: torch.Tensor,
    out_path: str,
    cmap: str = "turbo",
) -> None:
    """seq_x/seq_y/pred: (T,C,H,W)，C=1 雷达；保存三行×T 列（输入行只占用前 T_in 列，其余列可关掉）。"""
    tin, c, _, _ = seq_x.shape
    tout = seq_y.shape[0]
    assert c == 1 and pred.shape[0] == tout
    x = seq_x[:, 0].cpu().numpy()
    y = seq_y[:, 0].cpu().numpy()
    p = pred[:, 0].cpu().numpy()

    ncols = max(tin, tout)
    fig, axes = plt.subplots(3, ncols, figsize=(min(2.0 * ncols, 36), 6.5))
    if ncols == 1:
        axes = np.array(axes).reshape(3, 1)
    vmin, vmax = 0.0, 1.0

    for j in range(ncols):
        for i in range(3):
            axes[i, j].axis("off")

    for j in range(tin):
        axes[0, j].imshow(x[j], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, j].set_title(f"输入 {j + 1}/{tin}", fontsize=8)
        axes[0, j].axis("off")
    for j in range(tin, ncols):
        axes[0, j].axis("off")

    for j in range(tout):
        axes[1, j].imshow(y[j], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, j].set_title(f"真值 +{j + 1}", fontsize=8)
        axes[1, j].axis("off")
        axes[2, j].imshow(p[j], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[2, j].set_title(f"预报 +{j + 1}", fontsize=8)
        axes[2, j].axis("off")
    for j in range(tout, ncols):
        axes[1, j].axis("off")
        axes[2, j].axis("off")

    axes[0, 0].set_ylabel("输入", fontsize=10)
    axes[1, 0].set_ylabel("真值", fontsize=10)
    axes[2, 0].set_ylabel("预报", fontsize=10)
    plt.suptitle("归一化强度 [0,1]（与训练时 /255 一致）", fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="SimVP 雷达验证集个例推理与绘图")
    parser.add_argument("--weight_path", type=str, required=True, help="Lightning ckpt")
    parser.add_argument(
        "--val_file",
        type=str,
        default="preprocess/huadong_cr_6min/data_list_val.csv",
        help="与 train.py --val_file 相同（多列 PNG 或事件索引 CSV）",
    )
    parser.add_argument(
        "--radar_dir",
        type=str,
        required=True,
        help="验证样本根目录；多列 PNG 时为含相对路径的雷达根；事件模式为 val split 根（含 dBZ/）",
    )
    parser.add_argument(
        "--packed_event_npy",
        type=int,
        default=0,
        help="1：事件帧 data_dir_*/frame_*.npy；0：多列 PNG CSV",
    )
    parser.add_argument(
        "--packed_sequence_file",
        type=int,
        default=0,
        help="1：单列条带 PNG；与 packed_event_npy 互斥",
    )
    parser.add_argument(
        "--val_indices",
        type=int,
        nargs="+",
        default=[0, 1],
        help="val_file 中的行号（0-based），与 SingleRadarDataset 一致。"
        "若训练时 val_sample_interval>1，验证子集为第 0、N、2N… 行，请按需传例如 0 3。",
    )
    parser.add_argument("--output_dir", type=str, default=_DEFAULT_OUT)
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=("gpu", "npu", "cpu"),
        help="推理设备：gpu / 昇腾 npu / cpu（NPU 需 torch_npu + CANN 环境）",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="设备序号，优先于 --gpus；NPU 上如 0、1 表示 npu:0、npu:1",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="兼容 Lightning 写法：单卡填 0；多卡字符串取第一张，如 0,1",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser = SimVP_Lit.add_model_specific_args(parser)
    args = parser.parse_args()

    if int(args.packed_event_npy) and int(args.packed_sequence_file):
        raise SystemExit("packed_event_npy 与 packed_sequence_file 不能同时为 1")

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 勿把 argparse 里的模型默认值 ** 进 load_from_checkpoint：会覆盖 ckpt 内保存的
    # input_class / hid_S 等，导致与训练不一致（例如 ckpt 为单通道 hid_S=32，默认却是双通道 hid_S=64）。
    # 架构与训练超参一律以 checkpoint 为准；此处仅覆盖推理常用项。
    model = SimVP_Lit.load_from_checkpoint(
        args.weight_path,
        strict=False,
        map_location="cpu",
        batch_size=int(args.batch_size),
        test_save_path=args.test_save_path,
    )
    hp = model.hparams
    dataset = SingleRadarDataset(
        args.val_file,
        args.radar_dir,
        input_length=int(hp.input_length),
        target_length=int(hp.target_length),
        height=int(hp.height),
        width=int(hp.width),
        packed_sequence_file=bool(int(args.packed_sequence_file)),
        packed_event_npy=bool(int(args.packed_event_npy)),
    )

    n_all = len(dataset)
    indices: List[int] = []
    for i in args.val_indices:
        if i < 0 or i >= n_all:
            raise SystemExit(f"val_indices 含非法索引 {i}，验证集长度={n_all}")
        indices.append(int(i))

    device = _resolve_device(args.accelerator, args.devices, args.gpus)
    model.to(device)
    model.eval()

    subset = Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=min(int(args.batch_size), len(indices)),
        shuffle=False,
        num_workers=0,
    )

    records = []
    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            seqs_x, seqs_y = batch
            seqs_x = seqs_x.to(device)
            seqs_y = seqs_y.to(device)
            pred = model(seqs_x)
            pred = torch.clip(pred, 0, 1)

            b = seqs_x.shape[0]
            for k in range(b):
                global_idx = indices[batch_i * loader.batch_size + k]
                sx = seqs_x[k : k + 1]
                sy = seqs_y[k : k + 1]
                pr = pred[k : k + 1]
                mse = float(torch.mean((pr - sy) ** 2).item())
                mae = float(torch.mean(torch.abs(pr - sy)).item())
                records.append({"val_index": global_idx, "mse": mse, "mae": mae})
                plot_sample_triple(
                    sx[0],
                    sy[0],
                    pr[0],
                    os.path.join(out_dir, f"sample_validx_{global_idx:05d}.png"),
                )
                np.save(
                    os.path.join(out_dir, f"sample_validx_{global_idx:05d}_pred.npy"),
                    pr[0].cpu().numpy(),
                )
                np.save(
                    os.path.join(out_dir, f"sample_validx_{global_idx:05d}_gt.npy"),
                    sy[0].cpu().numpy(),
                )

    meta = {
        "weight_path": os.path.abspath(args.weight_path),
        "val_file": os.path.abspath(args.val_file),
        "radar_dir": os.path.abspath(args.radar_dir),
        "val_indices": indices,
        "n_val_total": n_all,
        "per_sample": records,
    }
    with open(os.path.join(out_dir, "infer_summary.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
