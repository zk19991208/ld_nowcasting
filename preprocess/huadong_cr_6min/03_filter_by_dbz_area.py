# 按「dBZ≥THRESH_DBZ 的像素面积占比」筛选 data_list 中的序列（剔除晴空/过弱回波）；PNG 为 0–70 dBZ 线性映射到 0–255。
# 对每帧计算占比后，按 AGGREGATE（max/mean）跨帧聚合，再与 MIN_FRAC_ABOVE_THRESH 比较；默认 max、面积占比 1%。
# 样本量大时用多进程并行处理「每条序列」，并显示 tqdm 进度。
# 运行：python preprocess/huadong_cr_6min/03_filter_by_dbz_area.py
# 修改路径与阈值、并行数：编辑下方「配置常量」。依赖：numpy、imageio、tqdm。
#
# 耗时粗估：瓶颈在读 PNG 与解码；单条约 (INPUT+TARGET) 帧×550×550（如 40 帧），机械盘约 0.3～4 s/条，
# SSD 约 0.1～1 s/条（量级仅供参考）。总时间 ≈ (样本数 / NUM_WORKERS) × 单条耗时 + 进程开销。

from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 配置常量（需与 01/02 中 ROOT_DIR 一致）
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent

ROOT_DIR = Path("/home/user/data/huadong/CR_6min_550x550")
DATA_LIST_CSV = _THIS_DIR / "data_list.csv"
OUT_CSV = _THIS_DIR / "data_list_clean.csv"

# 物理：像素值 0–255 对应 0–70 dBZ
DBZ_MAX = 70.0
THRESH_DBZ = 30.0
# 序列保留条件：聚合后的「≥THRESH_DBZ 像素面积占比」≥ 该值（默认 1%）
MIN_FRAC_ABOVE_THRESH = 0.01
# 跨帧聚合："max" 取各帧占比的最大值（不易误删短时强回波）；"mean" 更严
AGGREGATE = "max"

# 同上：多核 ARM/鲲鹏 勿默认拉满；读图+解码为 I/O 与内存敏感，进程数过多常变慢。
NUM_WORKERS_CAP = 32
NUM_WORKERS = min(NUM_WORKERS_CAP, max(1, (os.cpu_count() or 8) - 1))
MIN_ROWS_FOR_PARALLEL = 32
CHUNKSIZE = 8


def _load_gray_f32(path: Path) -> np.ndarray:
    arr = imageio.imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def _frame_frac_above_thresh(gray: np.ndarray) -> float:
    dbz = gray / 255.0 * DBZ_MAX
    return float(np.mean(dbz >= THRESH_DBZ))


def _aggregate(fracs: list[float]) -> float:
    if not fracs:
        return 0.0
    if AGGREGATE == "mean":
        return float(sum(fracs) / len(fracs))
    return max(fracs)


def _eval_one_row(payload: tuple[str, tuple[str, ...]]) -> list[str] | None:
    """子进程：满足阈值则返回该行路径列表，否则 None（读失败或占比不足）。"""
    root_str, rels = payload
    root = Path(root_str)
    fracs: list[float] = []
    for rel in rels:
        fp = root / rel
        if not fp.is_file():
            return None
        try:
            g = _load_gray_f32(fp)
            fracs.append(_frame_frac_above_thresh(g))
        except Exception:
            return None
    score = _aggregate(fracs)
    if score >= MIN_FRAC_ABOVE_THRESH:
        return list(rels)
    return None


def main() -> None:
    if not DATA_LIST_CSV.is_file():
        raise SystemExit(f"找不到 DATA_LIST_CSV: {DATA_LIST_CSV}")
    if not ROOT_DIR.is_dir():
        raise SystemExit(f"ROOT_DIR 不存在: {ROOT_DIR}")

    rows_in: list[tuple[str, ...]] = []
    with DATA_LIST_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all(not c.strip() for c in row):
                continue
            rels = tuple(c.strip().replace("\\", "/") for c in row)
            rows_in.append(rels)

    total = len(rows_in)
    root_str = str(ROOT_DIR.resolve())
    kept: list[list[str]] = []

    if total == 0:
        pass
    elif total < MIN_ROWS_FOR_PARALLEL:
        for rels in tqdm(rows_in, desc="筛选序列(单进程)", unit="seq"):
            r = _eval_one_row((root_str, rels))
            if r is not None:
                kept.append(r)
    else:
        payloads = ((root_str, rels) for rels in rows_in)
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
            it = ex.map(_eval_one_row, payloads, chunksize=CHUNKSIZE)
            for res in tqdm(
                it,
                total=total,
                desc=f"筛选序列({NUM_WORKERS}进程)",
                unit="seq",
            ):
                if res is not None:
                    kept.append(res)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp, lineterminator="\n")
        w.writerows(kept)

    print(
        f"写入 {OUT_CSV}：保留 {len(kept)} / 输入 {total} 条 "
        f"（MIN_FRAC_ABOVE_THRESH={MIN_FRAC_ABOVE_THRESH}, THRESH_DBZ={THRESH_DBZ}, AGGREGATE={AGGREGATE!r}）"
    )


if __name__ == "__main__":
    main()
