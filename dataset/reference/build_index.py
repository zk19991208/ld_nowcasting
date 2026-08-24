"""Scan NJU_CPOL data directory and build train/val/test index CSV files.

Usage:
    python build_index.py --data_root /liyang/data/NJU_CPOL \
                          --output_dir ./index \
                          --n_input 10 --n_output 20 \
                          --seed 42
"""

import argparse
import csv
import json
import os
import random
from glob import glob


def count_frames(event_dir: str) -> int:
    """Count .npy frame files in an event directory."""
    return len(glob(os.path.join(event_dir, "frame_*.npy")))


def build_index(
    data_root: str,
    output_dir: str,
    n_input: int = 10,
    n_output: int = 20,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    window_size = n_input + n_output
    dbz_root = os.path.join(data_root, "dBZ")

    event_dirs = sorted(glob(os.path.join(dbz_root, "data_dir_*")))
    if not event_dirs:
        raise FileNotFoundError(f"No event dirs found under {dbz_root}")

    event_info = []
    for edir in event_dirs:
        eid = int(os.path.basename(edir).split("_")[-1])
        n_frames = count_frames(edir)
        if n_frames >= window_size:
            n_samples = n_frames - window_size + 1
            event_info.append((eid, n_frames, n_samples))

    print(f"Total events: {len(event_dirs)}, "
          f"usable (>= {window_size} frames): {len(event_info)}")

    rng = random.Random(seed)
    eids = [e[0] for e in event_info]
    rng.shuffle(eids)

    n_total = len(eids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_events = set(eids[:n_train])
    val_events = set(eids[n_train : n_train + n_val])
    test_events = set(eids[n_train + n_val :])

    splits = {"train": [], "val": [], "test": []}
    eid_to_info = {e[0]: e for e in event_info}

    for eid, n_frames, n_samples in event_info:
        if eid in train_events:
            split = "train"
        elif eid in val_events:
            split = "val"
        else:
            split = "test"
        for start in range(n_samples):
            splits[split].append((eid, start))

    os.makedirs(output_dir, exist_ok=True)

    for split_name, samples in splits.items():
        csv_path = os.path.join(output_dir, f"{split_name}_index.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["event_id", "start_frame"])
            for eid, start in samples:
                writer.writerow([eid, start])
        print(f"  {split_name}: {len(samples)} samples -> {csv_path}")

    meta = {
        "data_root": data_root,
        "n_input_frames": n_input,
        "n_output_frames": n_output,
        "window_size": window_size,
        "frame_interval_minutes": 6,
        "total_events": len(event_dirs),
        "usable_events": len(event_info),
        "skipped_events": len(event_dirs) - len(event_info),
        "seed": seed,
        "split_ratio": {
            "train": train_ratio,
            "val": val_ratio,
            "test": round(1.0 - train_ratio - val_ratio, 4),
        },
        "split_events": {
            "train": sorted(train_events),
            "val": sorted(val_events),
            "test": sorted(test_events),
        },
        "split_samples": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
    }
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta -> {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Build radar nowcast index files")
    parser.add_argument("--data_root", type=str, default="/liyang/data/NJU_CPOL")
    parser.add_argument("--output_dir", type=str, default="./index")
    parser.add_argument("--n_input", type=int, default=10)
    parser.add_argument("--n_output", type=int, default=20)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_index(
        data_root=args.data_root,
        output_dir=args.output_dir,
        n_input=args.n_input,
        n_output=args.n_output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
