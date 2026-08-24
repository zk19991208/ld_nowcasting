# 读取 timeseries.csv，在连续 6min 序列段上滑窗，生成 data_list.csv（每行 input+target 个相对路径，无表头）。
# 与 dataset/RainDataset.SingleRadarDataset 约定一致：每行逗号分隔路径，radar_dir 与 ROOT_DIR 拼接读图。
# 滑窗数量大时，用多进程并行做「窗口内文件是否均存在」校验，并显示 tqdm 进度。
# 运行：python preprocess/huadong_cr_6min/02_build_data_list.py
# 修改输入/输出路径、输入与预报帧数、并行数：编辑下方「配置常量」。
#
# 耗时粗估：瓶颈在磁盘 stat/存在性检查；窗口数约 W 时，机械盘可能数分钟～数十分钟，
# NVMe SSD 上常为「每 1e5 窗口数秒～数十秒」量级（与 CPU 核数、是否冷启动有关），仅供参考。

from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# 配置常量（需与 01_scan_timeseries.py 中 ROOT_DIR 一致）
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent

ROOT_DIR = Path("/home/user/data/huadong/CR_6min_550x550")
TIMESERIES_CSV = _THIS_DIR / "timeseries.csv"
OUT_CSV = _THIS_DIR / "data_list.csv"

# 外推：前 INPUT_FRAMES 帧为输入，后 TARGET_FRAMES 帧为预报目标（列顺序从左到右）
INPUT_FRAMES = 20
TARGET_FRAMES = 20
INTERVAL_MINUTES = 6

# 并行：窗口数低于该值时单进程（避免进程开销）；否则用多进程校验
MIN_WINDOWS_FOR_PARALLEL = 2000
# 高核数 CPU（如鲲鹏 920 96 核）勿用「核数−1」拉满进程：stat/读盘多为 I/O 瓶颈，进程过多
# 反抢带宽与内存。可按磁盘实测 tqdm 调大（本地 NVMe 可试 48～64；网络盘建议 8～16）。
NUM_WORKERS_CAP = 32
NUM_WORKERS = min(NUM_WORKERS_CAP, max(1, (os.cpu_count() or 8) - 1))
# imap chunksize，略大可减少进程间通信次数
CHUNKSIZE = 64


def _parse_datetime_cell(s: str) -> datetime:
    s = s.strip()
    if len(s) >= 16 and s[4] == "-" and s[10] in " T":
        return datetime.strptime(s[:16].replace("T", " "), "%Y-%m-%d %H:%M")
    return datetime.fromisoformat(s.replace("Z", "+00:00").split("+")[0].strip())


def _window_all_exist(payload: tuple[str, tuple[str, ...]]) -> tuple[str, ...] | None:
    """子进程：若窗口内路径均存在则返回 paths，否则 None。"""
    root_str, paths = payload
    root = Path(root_str)
    if all((root / p).is_file() for p in paths):
        return paths
    return None


def main() -> None:
    if not TIMESERIES_CSV.is_file():
        raise SystemExit(f"找不到 TIMESERIES_CSV: {TIMESERIES_CSV}")
    if not ROOT_DIR.is_dir():
        raise SystemExit(f"ROOT_DIR 不存在: {ROOT_DIR}")

    records: list[tuple[datetime, str]] = []
    with TIMESERIES_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("timeseries.csv 无表头，需含 datetime, rel_path")
        for row in reader:
            dt = _parse_datetime_cell(row["datetime"])
            rel = row["rel_path"].strip().replace("\\", "/")
            records.append((dt, rel))

    records.sort(key=lambda x: (x[0], x[1]))
    delta = timedelta(minutes=INTERVAL_MINUTES)
    need = INPUT_FRAMES + TARGET_FRAMES
    pending: list[tuple[str, ...]] = []

    n = len(records)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and records[j + 1][0] - records[j][0] == delta:
            j += 1
        segment = records[i : j + 1]
        seg_len = len(segment)
        for k in range(0, seg_len - need + 1):
            window = segment[k : k + need]
            paths = tuple(p for _, p in window)
            pending.append(paths)
        i = j + 1

    root_str = str(ROOT_DIR.resolve())
    out_rows: list[list[str]] = []

    if len(pending) == 0:
        pass
    elif len(pending) < MIN_WINDOWS_FOR_PARALLEL:
        for paths in tqdm(pending, desc="校验窗口(单进程)", unit="win"):
            if _window_all_exist((root_str, paths)):
                out_rows.append(list(paths))
    else:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
            # ProcessPoolExecutor 仅有 map，无 imap（imap 在 multiprocessing.Pool）
            it = ex.map(
                _window_all_exist,
                ((root_str, p) for p in pending),
                chunksize=CHUNKSIZE,
            )
            for res in tqdm(
                it,
                total=len(pending),
                desc=f"校验窗口({NUM_WORKERS}进程)",
                unit="win",
            ):
                if res is not None:
                    out_rows.append(list(res))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        for paths in out_rows:
            w.writerow(paths)

    print(f"写入 {OUT_CSV}，共 {len(out_rows)} 条序列（每行 {need} 列），待校验窗口 {len(pending)} 个。")


if __name__ == "__main__":
    main()
