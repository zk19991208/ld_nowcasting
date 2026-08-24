# 扫描华东 CR 6min 雷达 PNG 目录树，生成按时间排序的 timeseries.csv（datetime + 相对路径）。
# 目录约定：ROOT_DIR / {year} / {yyyymmdd} / {yyyymmddHHMM}.png
# 运行（在仓库根目录或本目录，无需参数）：python preprocess/huadong_cr_6min/01_scan_timeseries.py
# 修改数据根目录、日期范围、输出路径：编辑下方「配置常量」。

from __future__ import annotations

import csv
import warnings
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 配置常量（按需修改）
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent

# 数据根目录（示例：华东 CR 6min 550x550）
ROOT_DIR = Path("/home/user/data/huadong/CR_6min_550x550")

# 输出 CSV（含表头：datetime, rel_path）
OUT_CSV = _THIS_DIR / "timeseries.csv"

# 仅保留该时间范围内的文件（含起、含止日全天）
DATE_START = datetime(2023, 5, 1, 0, 0, 0)
# 结束采用「不含」上界：早于该时刻的样本均保留，即有效至 2025-10-31 23:59 的最后一档 6min
DATE_END_EXCLUSIVE = datetime(2025, 11, 1, 0, 0, 0)

FILE_EXT = ".png"
WRITE_HEADER = True


def _parse_dt_from_stem(stem: str) -> datetime | None:
    """文件名主干 12 位 yyyymmddHHMM -> naive datetime。"""
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


def _rel_path_posix(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def main() -> None:
    root = ROOT_DIR
    if not root.is_dir():
        raise SystemExit(f"ROOT_DIR 不存在或不是目录: {root}")

    rows: list[tuple[datetime, str]] = []
    pattern = f"*{FILE_EXT}"
    for png in sorted(root.rglob(pattern)):
        if not png.is_file():
            continue
        try:
            rel = _rel_path_posix(root, png)
        except ValueError:
            continue
        parts = Path(rel).parts
        if len(parts) != 3:
            continue
        year_s, ymd_dir, fname = parts
        stem = Path(fname).stem
        dt = _parse_dt_from_stem(stem)
        if dt is None:
            warnings.warn(f"无法解析时间，跳过: {rel}")
            continue
        if stem[:4] != year_s or stem[:8] != ymd_dir:
            warnings.warn(f"目录与文件名日期不一致，跳过: {rel}")
            continue
        if not (DATE_START <= dt < DATE_END_EXCLUSIVE):
            continue
        rows.append((dt, rel))

    # 按时间排序；同一时间只保留一条（保留首次出现路径）
    seen: set[datetime] = set()
    deduped: list[tuple[datetime, str]] = []
    for dt, rel in sorted(rows, key=lambda x: (x[0], x[1])):
        if dt in seen:
            continue
        seen.add(dt)
        deduped.append((dt, rel))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if WRITE_HEADER:
            w.writerow(["datetime", "rel_path"])
        for dt, rel in deduped:
            w.writerow([dt.strftime("%Y-%m-%d %H:%M"), rel])

    print(f"写入 {OUT_CSV}，共 {len(deduped)} 条（去重后）。")


if __name__ == "__main__":
    main()
