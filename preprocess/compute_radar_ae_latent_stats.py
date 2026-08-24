"""Compute training-set channel statistics for a frozen radar AutoEncoder.

Run from the project root before training the latent Predictor. The same frame
sampling distribution as the sequence training CSV is used; validation/test data
must not be supplied here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset.RainDataset import SingleRadarDataset
from models.radar_autoencoder import RadarAutoEncoderLit


def _device(value: str) -> torch.device:
    value = value.lower().strip()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {value}, but torch reports no CUDA/HIP device")
    return torch.device(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="统计冻结雷达 AE 的训练集 latent 均值/标准差")
    parser.add_argument("--ae_checkpoint", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--radar_dir", required=True)
    parser.add_argument("--output", required=True, help="输出 .npz")
    parser.add_argument("--input_length", type=int, default=20)
    parser.add_argument("--target_length", type=int, default=20)
    parser.add_argument("--height", type=int, default=560)
    parser.add_argument("--width", type=int, default=560)
    parser.add_argument("--packed_sequence_file", type=int, choices=(0, 1), default=0)
    parser.add_argument("--packed_event_npy", type=int, choices=(0, 1), default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--encode_chunk_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=0, help="0 表示扫描完整训练集")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.packed_sequence_file and args.packed_event_npy:
        parser.error("packed_sequence_file and packed_event_npy cannot both be 1")
    if args.batch_size <= 0 or args.encode_chunk_size <= 0:
        parser.error("batch_size and encode_chunk_size must be positive")

    device = _device(args.device)
    ae = RadarAutoEncoderLit.load_from_checkpoint(args.ae_checkpoint, map_location="cpu")
    ae.eval().requires_grad_(False).to(device)
    if (int(ae.hparams.height), int(ae.hparams.width)) != (args.height, args.width):
        raise ValueError(
            "requested H/W differs from AE checkpoint: "
            f"{(args.height, args.width)} vs {(ae.hparams.height, ae.hparams.width)}"
        )

    dataset = SingleRadarDataset(
        args.train_file,
        args.radar_dir,
        input_length=args.input_length,
        target_length=args.target_length,
        height=args.height,
        width=args.width,
        packed_sequence_file=bool(args.packed_sequence_file),
        packed_event_npy=bool(args.packed_event_npy),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    channels = int(ae.hparams.ae_latent_channels)
    channel_sum = torch.zeros(channels, dtype=torch.float64)
    channel_square_sum = torch.zeros(channels, dtype=torch.float64)
    count = 0
    batches_seen = 0

    with torch.inference_mode():
        for batch_index, (seqs_x, seqs_y) in enumerate(loader):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            sequence = torch.cat((seqs_x, seqs_y), dim=1).float()
            flat = sequence.flatten(0, 1)
            for frames in flat.split(args.encode_chunk_size, dim=0):
                latent = ae.encode(frames.to(device, non_blocking=True)).float().cpu().double()
                channel_sum += latent.sum(dim=(0, 2, 3))
                channel_square_sum += latent.square().sum(dim=(0, 2, 3))
                count += int(latent.shape[0] * latent.shape[2] * latent.shape[3])
            batches_seen += 1
            if batches_seen % 50 == 0:
                print(f"processed batches={batches_seen}, samples={min(len(dataset), batches_seen * args.batch_size)}")

    if count == 0:
        raise RuntimeError("no latent values were processed")
    mean = channel_sum / count
    variance = channel_square_sum / count - mean.square()
    std = variance.clamp_min(0.0).sqrt().clamp_min(1e-6)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        mean=mean.numpy().astype(np.float32),
        std=std.numpy().astype(np.float32),
        count=np.asarray(count, dtype=np.int64),
        batches=np.asarray(batches_seen, dtype=np.int64),
    )
    summary = {
        "output": str(output),
        "count_per_channel": count,
        "batches": batches_seen,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
