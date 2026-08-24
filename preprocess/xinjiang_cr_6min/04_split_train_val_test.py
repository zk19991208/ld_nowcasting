# 按样本起始时刻（每行第一列 PNG 对应时间）将清洗后的序列列表划分为 train / val / test 三个无表头 CSV。
# 划分规则（左闭右开）：start_time < TRAIN_END -> train；TRAIN_END <= start_time < VAL_END -> val；start_time >= VAL_END -> test。
# 运行：python preprocess/huadong_cr_6min/04_split_train_val_test.py
# 修改划分边界与路径：编辑下方「配置常量」。输入应为 03_filter_by_dbz_area.py 输出的 data_list_clean.csv。

from __future__ import annotations

import csv
import warnings
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 配置常量
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent

DATA_LIST_CSV = _THIS_DIR / "data_list_clean.csv"
OUT_TRAIN = _THIS_DIR / "data_list_train.csv"
OUT_VAL = _THIS_DIR / "data_list_val.csv"
OUT_TEST = _THIS_DIR / "data_list_test.csv"

# 时间块边界（naive datetime，与文件名时间一致）
TRAIN_END = datetime(2025, 5, 1, 0, 0, 0)
VAL_END = datetime(2025, 8, 1, 0, 0, 0)


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


def _start_time_from_row(row: list[str]) -> datetime | None:
    if not row:
        return None
    first = row[0].strip()
    stem = Path(first.replace("\\", "/")).stem
    return _parse_dt_from_stem(stem)


def main() -> None:
    if not DATA_LIST_CSV.is_file():
        raise SystemExit(f"找不到 DATA_LIST_CSV: {DATA_LIST_CSV}")

    train_rows: list[list[str]] = []
    val_rows: list[list[str]] = []
    test_rows: list[list[str]] = []

    with DATA_LIST_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all(not c.strip() for c in row):
                continue
            st = _start_time_from_row(row)
            if st is None:
                warnings.warn(f"无法从首列解析起始时间，跳过: {row[0][:80]!r}")
                continue
            if st < TRAIN_END:
                train_rows.append(row)
            elif st < VAL_END:
                val_rows.append(row)
            else:
                test_rows.append(row)

    def _write(path: Path, rows: list[list[str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as fp:
            w = csv.writer(fp, lineterminator="\n")
            w.writerows(rows)

    _write(OUT_TRAIN, train_rows)
    _write(OUT_VAL, val_rows[::2])
    _write(OUT_TEST, test_rows)

    print(
        f"划分完成：train={len(train_rows)} -> {OUT_TRAIN.name}, "
        f"val={len(val_rows[::2])} -> {OUT_VAL.name}, "
        f"test={len(test_rows)} -> {OUT_TEST.name}"
    )
    if len(train_rows) == 0:
        warnings.warn("train 集合为空，请检查 TRAIN_END 与数据时间范围。")
    if len(val_rows[::2]) == 0:
        warnings.warn("val 集合为空。")
    if len(test_rows) == 0:
        warnings.warn("test 集合为空。")


if __name__ == "__main__":
    main()
