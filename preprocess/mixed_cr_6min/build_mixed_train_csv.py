# 训练：合并新疆 + 华东两份「多列 PNG 相对路径」训练 CSV，每列加区域子目录前缀。
# 验证/测试：仅从 xinjiang_xr_6min 读取列表，新疆为 val/test 区域，每列统一加新疆文件夹前缀，
# 供 SingleRadarDataModule 与混合训练共用同一 radar_dir（如 /home/usr/data）。
#
# 运行方式（在 transfer 项目根目录下）：
#   python preprocess/mixed_cr_6min/build_mixed_train_csv.py
# 修改下方「固定参数」区即可。

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

# ---------- 固定参数（按需要改这里）----------
_TRANSFER_ROOT = Path(__file__).resolve().parents[2]

# 训练：新疆 + 华东合并
XINJIANG_TRAIN_CSV = _TRANSFER_ROOT / "preprocess" / "xinjiang_cr_6min" / "data_list_train.csv"
HUADONG_TRAIN_CSV = _TRANSFER_ROOT / "preprocess" / "huadong_cr_6min" / "data_list_train.csv"
OUT_TRAIN_CSV = _TRANSFER_ROOT / "preprocess" / "mixed_cr_6min" / "data_list_train_mixed.csv"

# 验证/测试：仅新疆（xinjiang_xr_6min），列路径前加与训练一致的新疆子目录前缀
XINJIANG_XR_DIR = _TRANSFER_ROOT / "preprocess" / "xinjiang_cr_6min"
XINJIANG_VAL_CSV = XINJIANG_XR_DIR / "data_list_val.csv"
XINJIANG_TEST_CSV = XINJIANG_XR_DIR / "data_list_test.csv"
OUT_VAL_CSV = _TRANSFER_ROOT / "preprocess" / "mixed_cr_6min" / "data_list_val.csv"
OUT_TEST_CSV = _TRANSFER_ROOT / "preprocess" / "mixed_cr_6min" / "data_list_test.csv"

PREFIX_XJ = "xinjiang/CR_6min_550x550"
PREFIX_HD = "huadong/CR_6min_550x550"

EXPECT_COLS = 40  # input_length + target_length；与 train.py 不一致时改此处
SHUFFLE_TRAIN = False
SHUFFLE_SEED = 42
# -------------------------------------------


def _norm_prefix(p: str) -> str:
    p = p.strip().replace("\\", "/").strip("/")
    return p + "/" if p else ""


def _read_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open(newline="", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            parts = [c.strip() for c in line.split(",")]
            rows.append(parts)
    return rows


def _prefix_cells(cells: list[str], prefix: str) -> list[str]:
    pre = _norm_prefix(prefix)
    out: list[str] = []
    for c in cells:
        c = c.strip().replace("\\", "/").lstrip("/")
        out.append(f"{pre}{c}")
    return out


def _validate_ncols(rows: list[list[str]], path: Path, ncols: int) -> None:
    for i, row in enumerate(rows):
        if len(row) != ncols:
            print(f"错误: {path} 第 {i + 1} 行列数={len(row)}，期望 {ncols}", file=sys.stderr)
            sys.exit(1)


def _write_csv_rows(rows: list[list[str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp, lineterminator="\n")
        for row in rows:
            w.writerow(row)


def _build_train() -> None:
    xj_rows = _read_rows(XINJIANG_TRAIN_CSV)
    hd_rows = _read_rows(HUADONG_TRAIN_CSV)

    if not xj_rows and not hd_rows:
        print("错误: 训练两个 CSV 均无数据行", file=sys.stderr)
        sys.exit(1)

    ncols = len(xj_rows[0]) if xj_rows else len(hd_rows[0])
    _validate_ncols(xj_rows, XINJIANG_TRAIN_CSV, ncols)
    _validate_ncols(hd_rows, HUADONG_TRAIN_CSV, ncols)

    if ncols != EXPECT_COLS:
        print(f"错误: 训练列数={ncols} 与 EXPECT_COLS={EXPECT_COLS} 不符", file=sys.stderr)
        sys.exit(1)

    merged: list[list[str]] = []
    merged.extend(_prefix_cells(row, PREFIX_XJ) for row in xj_rows)
    merged.extend(_prefix_cells(row, PREFIX_HD) for row in hd_rows)

    if SHUFFLE_TRAIN:
        rng = random.Random(SHUFFLE_SEED)
        rng.shuffle(merged)

    _write_csv_rows(merged, OUT_TRAIN_CSV)
    print(
        f"[train] 写出 {OUT_TRAIN_CSV}：新疆 {len(xj_rows)} 行 + 华东 {len(hd_rows)} 行 => 共 {len(merged)} 行，每行 {ncols} 列"
    )


def _build_xr_split(src: Path, out: Path, label: str) -> None:
    if not src.is_file():
        print(f"跳过 {label}: 源文件不存在 {src}", file=sys.stderr)
        return
    rows = _read_rows(src)
    if not rows:
        print(f"错误: {label} {src} 无数据行", file=sys.stderr)
        sys.exit(1)
    ncols = len(rows[0])
    _validate_ncols(rows, src, ncols)
    if ncols != EXPECT_COLS:
        print(f"错误: {label} 列数={ncols} 与 EXPECT_COLS={EXPECT_COLS} 不符", file=sys.stderr)
        sys.exit(1)
    prefixed = [_prefix_cells(row, PREFIX_XJ) for row in rows]
    _write_csv_rows(prefixed, out)
    print(f"[{label}] 写出 {out}：{len(rows)} 行（前缀 {PREFIX_XJ}/）")


def main() -> None:
    _build_train()
    _build_xr_split(XINJIANG_VAL_CSV, OUT_VAL_CSV, "val")
    _build_xr_split(XINJIANG_TEST_CSV, OUT_TEST_CSV, "test")


if __name__ == "__main__":
    main()
