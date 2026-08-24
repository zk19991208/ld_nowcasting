# 将 04 输出的滑动窗口 CSV 按「相邻样本起始时刻相差 6 分钟」连成事件，每事件导出为 dataset/reference 风格目录：
#   {FRAME_ROOT}/{split}/dBZ/data_dir_{event_id:03d}/frame_{i:05d}.npy（uint8，H×W，与 PNG 灰度一致）。
# 并生成 data_list_*_events.csv（event_id,start_frame,start_time）与 events_manifest.json。
# 不再生成事件条带 PNG。
# 用法（在仓库 transfer 根目录）：
#   conda activate torch
#   python preprocess/huadong_cr_6min/05_export_event_frames_npy.py
# 修改下方 ROOT_DIR、FRAME_ROOT、SPLITS、FRAME_DELTA_MINUTES、NUM_WORKERS。

from __future__ import annotations

import csv
import json
import os
import sys
import multiprocessing as mp
from datetime import datetime, timedelta
from pathlib import Path

import imageio
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

_THIS_DIR = Path(__file__).resolve().parent

ROOT_DIR = Path("/home/user/data/huadong/CR_6min_550x550")

SPLITS = {
    "train": _THIS_DIR / "data_list_train.csv",
    "val": _THIS_DIR / "data_list_val.csv",
    "test": _THIS_DIR / "data_list_test.csv",
}

FRAME_ROOT = Path("/home/user/data/huadong/CR_6min_event_npy")

OUT_INDEX = {
    "train": _THIS_DIR / "data_list_train_events.csv",
    "val": _THIS_DIR / "data_list_val_events.csv",
    "test": _THIS_DIR / "data_list_test_events.csv",
}

META_JSON = _THIS_DIR / "event_export_meta.json"

NUM_WORKERS = 0

FRAME_DELTA_MINUTES = 6


def _parse_dt_from_stem(stem: str) -> datetime | None:
    if len(stem) != 12 or not stem.isdigit():
        return None
    try:
        return datetime(
            int(stem[0:4]),
            int(stem[4:6]),
            int(stem[6:8]),
            int(stem[8:10]),
            int(stem[10:12]),
            0,
        )
    except ValueError:
        return None


def _start_dt_from_path(rel: str) -> datetime:
    stem = Path(rel.replace("\\", "/")).stem
    dt = _parse_dt_from_stem(stem)
    if dt is None:
        raise ValueError(f"无法从路径解析时间: {rel!r}")
    return dt


def _load_rows(csv_path: Path) -> np.ndarray:
    arr = np.loadtxt(csv_path, delimiter=",", dtype=str)
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    return arr


def _sort_indices_by_start_time(rows: np.ndarray) -> np.ndarray:
    n = rows.shape[0]
    keys = []
    for i in range(n):
        keys.append((_start_dt_from_path(rows[i, 0]), i))
    keys.sort(key=lambda x: x[0])
    return np.array([i for _, i in keys], dtype=np.int64)


def _group_events(rows: np.ndarray) -> list[list[int]]:
    order = _sort_indices_by_start_time(rows)
    groups: list[list[int]] = []
    cur: list[int] = []
    prev_dt: datetime | None = None
    delta = timedelta(minutes=FRAME_DELTA_MINUTES)

    for j in range(len(order)):
        idx = int(order[j])
        dt = _start_dt_from_path(rows[idx, 0])
        if not cur:
            cur = [idx]
            prev_dt = dt
            continue
        if dt == prev_dt + delta:
            cur.append(idx)
            prev_dt = dt
        else:
            groups.append(cur)
            cur = [idx]
            prev_dt = dt
    if cur:
        groups.append(cur)
    return groups


def _merge_paths_for_event(rows: np.ndarray, row_indices: list[int]) -> list[str]:
    first = rows[row_indices[0]]
    paths = first.tolist()
    for k in range(1, len(row_indices)):
        r = rows[row_indices[k]]
        paths.append(r[-1])
    return paths


def _time_str_from_rel(rel: str) -> str:
    stem = Path(rel.replace("\\", "/")).stem
    return stem


def _frame_hw_from_rel(rel: str, root_dir: str) -> tuple[int, int]:
    p = os.path.join(root_dir, rel)
    im = np.asarray(imageio.imread(p))
    if im.ndim == 3:
        im = im[..., 0]
    h, w = im.shape[:2]
    return int(h), int(w)


def _export_one_event(task: tuple) -> int:
    eid, merged_paths, root_s, dbz_root_str = task
    ev_dir = os.path.join(dbz_root_str, f"data_dir_{eid:03d}")
    os.makedirs(ev_dir, exist_ok=True)
    for i, rel in enumerate(merged_paths):
        p = os.path.join(root_s, rel)
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        im = np.asarray(imageio.imread(p))
        if im.ndim == 3:
            im = im[..., 0]
        out_p = os.path.join(ev_dir, f"frame_{i:05d}.npy")
        np.save(out_p, im.astype(np.uint8, copy=False))
    return eid


def _resolve_num_workers() -> int:
    if NUM_WORKERS <= 0:
        return max(1, min(8, (mp.cpu_count() or 2) - 1))
    return max(1, NUM_WORKERS)


def _process_split(split: str, csv_in: Path, frame_root: Path) -> dict:
    rows = _load_rows(csv_in)
    ncols = rows.shape[1]
    groups = _group_events(rows)

    dbz_root = frame_root / split / "dBZ"
    dbz_root.mkdir(parents=True, exist_ok=True)

    index_rows: list[tuple[int, int, str]] = []
    tasks = []
    manifest: dict[str, dict[str, int]] = {}
    root_s = str(ROOT_DIR)
    dbz_s = str(dbz_root)

    for eid, idx_list in enumerate(groups):
        merged = _merge_paths_for_event(rows, idx_list)
        fh, fw = _frame_hw_from_rel(merged[0], root_s)
        manifest[str(eid)] = {
            "n_frames": len(merged),
            "frame_height": fh,
            "frame_width": fw,
        }
        tasks.append((eid, merged, root_s, dbz_s))
        for j in range(len(idx_list)):
            start_rel = rows[idx_list[j], 0]
            tstr = _time_str_from_rel(start_rel)
            index_rows.append((eid, j, tstr))

    meta = {
        "split": split,
        "source_csv": str(csv_in),
        "num_source_rows": int(rows.shape[0]),
        "frames_per_window": int(ncols),
        "num_events": len(groups),
        "num_samples": len(index_rows),
        "frame_interval_minutes": FRAME_DELTA_MINUTES,
    }

    nw = _resolve_num_workers()
    desc = f"[{split}] 写事件帧 .npy"
    if nw <= 1:
        for t in tqdm(tasks, desc=desc, unit="事件", ncols=100):
            _export_one_event(t)
    else:
        chunksize = max(1, len(tasks) // (nw * 8))
        process_map(_export_one_event, tasks, max_workers=nw, chunksize=chunksize, desc=desc, unit="事件", ncols=100)

    out_csv = OUT_INDEX[split]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "start_frame", "start_time"])
        for eid, sf, tst in index_rows:
            w.writerow([eid, sf, tst])

    split_root = frame_root / split
    man_path = split_root / "events_manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(
        f"[{split}] 事件数={len(groups)}, 样本数={len(index_rows)} "
        f"(原行数={rows.shape[0]}) -> {out_csv} + {man_path}",
        flush=True,
    )
    return meta


def main() -> None:
    if not ROOT_DIR.is_dir():
        print(f"错误: ROOT_DIR 不存在: {ROOT_DIR}", file=sys.stderr)
        sys.exit(1)

    frame_abs = FRAME_ROOT
    frame_abs.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("05_export_event_frames_npy — 事件帧目录 dBZ/data_dir_*/frame_*.npy + 索引 CSV")
    print(f"  读图根目录 ROOT_DIR: {ROOT_DIR}")
    print(f"  输出根 FRAME_ROOT: {frame_abs}")
    print(f"  相邻行合并间隔: {FRAME_DELTA_MINUTES} 分钟")
    print("=" * 60)

    all_meta: dict = {}
    done = 0
    for split, csv_in in SPLITS.items():
        if not csv_in.is_file():
            print(f"\n[跳过] 列表不存在: {csv_in}")
            continue
        all_meta[split] = _process_split(split, csv_in, frame_abs)
        done += 1

    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ROOT_DIR": str(ROOT_DIR),
                "FRAME_ROOT": str(frame_abs),
                "FRAME_DELTA_MINUTES": FRAME_DELTA_MINUTES,
                "splits": all_meta,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nmeta -> {META_JSON}")
    print("\n" + "=" * 60)
    print(f"全部结束: 已处理 {done} 个 split。训练请使用 --packed_event_npy 1，")
    print(f"radar_dir 设为各 split 目录，例如: {frame_abs}/train（其下含 dBZ/ 与 events_manifest.json）")
    print("=" * 60)


if __name__ == "__main__":
    main()
