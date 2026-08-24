# 华东 CR 6min 雷达回波外推数据预处理

本目录提供五步脚本（前四步为必须，第五步为可选）：从按年月日归档的 PNG 扫描时间序列，生成连续帧样本列表，**按 dBZ 面积占比清洗**后，再按时间块划分训练/验证/测试集；可选将滑动窗口按**连续时间**合并为事件，并导出为 **dataset/reference 风格**的每帧 `.npy` 目录与 `event_id,start_frame` 索引（训练时只读窗口长度 T 帧，避免整段条带 PNG 解码）。**所有路径与超参均在各 `.py` 文件顶部常量中配置，运行时不传命令行参数。**

## 数据目录约定

- 根目录（示例）：`/home/user/data/huadong/CR_6min_550x550`
- 相对路径（POSIX）：`2023/20230501/202305010030.png`
  - 即 `{year}/{yyyymmdd}/{yyyymmddHHMM}.png`，文件名 12 位为观测时间（本地 naive 时间）。
- 时间步长：6 分钟。
- PNG 灰度值：业务上为 **0–70 dBZ 线性映射到 0–255**（清洗脚本 `03` 中按此还原 dBZ）。
- 计划内时间范围：2023-05-01 起至 2025-10-31 止（在 `01_scan_timeseries.py` 中由 `DATE_START` / `DATE_END_EXCLUSIVE` 控制）。

## 执行顺序

在仓库根目录下（已配置好 Python 环境，需安装 `numpy`、`imageio`、`tqdm`）依次执行：

```text
python preprocess/huadong_cr_6min/01_scan_timeseries.py
python preprocess/huadong_cr_6min/02_build_data_list.py
python preprocess/huadong_cr_6min/03_filter_by_dbz_area.py
python preprocess/huadong_cr_6min/04_split_train_val_test.py
python preprocess/huadong_cr_6min/05_export_event_frames_npy.py
```

可选第 5 步：将 `04` 产出的 `data_list_*.csv` 中**起始时刻连续相差 6 分钟**的滑动窗口合并为事件，并写入 **`{FRAME_ROOT}/{split}/dBZ/data_dir_{eid}/frame_{i:05d}.npy`**（uint8，与 PNG 灰度一致），同时生成 **`data_list_*_events.csv`**（表头 `event_id,start_frame,start_time`）及各 split 下的 **`events_manifest.json`**。训练时 **`radar_dir` 指向各 split 根目录**（如 `FRAME_ROOT/train`），并设置 **`--packed_event_npy 1`**，可选 **`radar_dir_val` / `radar_dir_test`**。

Windows 下将各脚本内的 `ROOT_DIR` 等改为你的本机路径即可。

## 各脚本常量说明

| 脚本 | 主要常量 | 含义 |
|------|----------|------|
| `01_scan_timeseries.py` | `ROOT_DIR` | 雷达 PNG 根目录 |
| | `OUT_CSV` | 输出 `timeseries.csv` 路径（默认本目录下） |
| | `DATE_START`, `DATE_END_EXCLUSIVE` | 扫描时间过滤区间（左闭右开） |
| | `FILE_EXT`, `WRITE_HEADER` | 扩展名、是否写表头 |
| `02_build_data_list.py` | `ROOT_DIR` | 与上一步一致，用于检查文件是否存在 |
| | `TIMESERIES_CSV`, `OUT_CSV` | 输入/输出的 CSV 路径 |
| | `INPUT_FRAMES`, `TARGET_FRAMES` | 输入帧数、预报帧数（每行共 `INPUT_FRAMES + TARGET_FRAMES` 列） |
| | `INTERVAL_MINUTES` | 固定为 6 |
| | `NUM_WORKERS_CAP`, `NUM_WORKERS`, `MIN_WINDOWS_FOR_PARALLEL`, `CHUNKSIZE` | 多进程校验窗口：实际进程数为 `min(NUM_WORKERS_CAP, CPU−1)`，避免鲲鹏/高核数拉满抢盘；其余为单进程阈值与 `imap` 块大小 |
| `03_filter_by_dbz_area.py` | `DATA_LIST_CSV` | 输入 `data_list.csv`（02 的输出） |
| | `OUT_CSV` | 输出 `data_list_clean.csv` |
| | `ROOT_DIR` | 读图根目录（与 01/02 一致） |
| | `DBZ_MAX`, `THRESH_DBZ` | 线性标定上限（70）、反射率阈值 dBZ（脚本内可改，不限定 25） |
| | `MIN_FRAC_ABOVE_THRESH` | **默认 `0.01`**：各帧「像素 dBZ≥`THRESH_DBZ`」占比经 `AGGREGATE` 聚合后须 ≥ 该值才保留 |
| | `AGGREGATE` | `"max"` 或 `"mean"`：跨帧聚合方式（默认 `max`） |
| | `NUM_WORKERS_CAP`, `NUM_WORKERS`, `MIN_ROWS_FOR_PARALLEL`, `CHUNKSIZE` | 多进程按「每条序列」并行：实际 `NUM_WORKERS=min(NUM_WORKERS_CAP, CPU−1)`；样本数低于 `MIN_ROWS_FOR_PARALLEL` 时走单进程 |
| `04_split_train_val_test.py` | `DATA_LIST_CSV` | **清洗后的** `data_list_clean.csv` |
| | `OUT_TRAIN`, `OUT_VAL`, `OUT_TEST` | 划分后的三个输出路径 |
| | `TRAIN_END`, `VAL_END` | 时间块边界（见下节） |
| `05_export_event_frames_npy.py` | `ROOT_DIR` | 与 PNG 根目录一致，用于读原图并写每帧 `.npy` |
| | `SPLITS` / `FRAME_ROOT` / `OUT_INDEX` | 输入 `data_list_*.csv`；输出 `{FRAME_ROOT}/{split}/dBZ/data_dir_*/frame_*.npy` 与 `events_manifest.json`；索引 CSV 为 `OUT_INDEX` |
| | `FRAME_DELTA_MINUTES` | 相邻样本起始时刻间隔，默认 6（须与数据一致） |
| | `NUM_WORKERS` | `0` 自动 `min(8, CPU−1)` 多进程；`1` 单进程 |

## 输出文件格式

1. **`timeseries.csv`**：表头 `datetime,rel_path`；`datetime` 形如 `2023-05-01 00:30`。
2. **`data_list.csv`**：**无表头**；每行一条训练样本，列为按时间顺序的相对路径，逗号分隔。
3. **`data_list_clean.csv`**：格式同上，为通过 **`THRESH_DBZ` 与 `MIN_FRAC_ABOVE_THRESH`（默认 ≥1% 像素面积）** 筛选后的子集。
4. **`data_list_train.csv` / `data_list_val.csv` / `data_list_test.csv`**：格式与 `data_list.csv` 相同，无表头。

### 划分规则（`04_split_train_val_test.py`）

以每行**第一列** PNG 对应的起始时刻 `start_time` 为准（由文件名 12 位解析）：

- **train**：`start_time < TRAIN_END`
- **val**：`TRAIN_END <= start_time < VAL_END`
- **test**：`start_time >= VAL_END`

默认：`TRAIN_END = 2025-06-01 00:00`，`VAL_END = 2025-09-01 00:00`。可按需要修改常量。

## 与训练代码的衔接

[`dataset/RainDataset.py`](../../dataset/RainDataset.py) 中 `SingleRadarDataset` 支持：

- **PNG 模式（默认）**：`csv` 每行为逗号分隔的**相对路径**；**`input_length` / `target_length` 须与预处理 `INPUT_FRAMES` / `TARGET_FRAMES` 一致**（例如各为 20，则每行共 40 列）。`radar_dir` 与预处理 **`ROOT_DIR`** 相同。
- **逐样本条带模式**：CSV 每行一个横向条带 PNG 相对路径，训练参数 **`packed_sequence_file: 1`**；条带在内存中切回 `(T, H, W)`。
- **事件帧目录模式**：使用 `05` 生成的 **`data_list_*_events.csv`**，`radar_dir` 指向 **`{FRAME_ROOT}/{split}`**（含 `dBZ/` 与 `events_manifest.json`），并设置 **`packed_event_npy: 1`**（或 `--packed_event_npy 1`），可选 **`radar_dir_val` / `radar_dir_test`**。

- 建议使用经 `03` 清洗、`04` 划分后的 **`data_list_train.csv` 等**；若已跑 `05` 导出事件帧，则训练列表改用 **`data_list_*_events.csv`** 与上述参数。
- **ConvLSTM 示例配置（YAML）**：[../../ascend_run/configs/convlstm_huadong_cr_550.yaml](../../ascend_run/configs/convlstm_huadong_cr_550.yaml)。在仓库根目录执行：`cd ascend_run && python run_train_yaml.py --config configs/convlstm_huadong_cr_550.yaml`（先按该文件内注释修改 `radar_dir` 与数据路径）。

## 说明

- 若磁盘上存在时间缺口，连续段会变短，滑窗样本数会减少，属正常现象。
- 大数据量时扫描可能较慢，可在 `01` 中按需改为按年遍历等优化（当前为递归 `rglob`）。

### 性能、多进程与 tqdm

- `02`、`03` 在耗时阶段会显示 **tqdm 进度条**；窗口数或序列数较少时自动退回单进程，避免进程启停开销。
- **`02_build_data_list.py`**：主要耗时在「每个滑窗内 `(INPUT_FRAMES+TARGET_FRAMES)` 个路径的 `exists`/stat」（如 20+20 则为 40 个）。粗估：与**窗口总数**及磁盘 IOPS 成正比；机械盘上可能达数分钟～数十分钟，NVMe 上常为「每约 10⁵ 窗口数秒～数十秒」量级（仅经验值，与是否冷缓存、CPU 核数有关）。总时间可近似为 `(窗口数 / 并行度) × 单次 stat 成本`，并行度受 `NUM_WORKERS` 与磁盘上限影响。
- **`03_filter_by_dbz_area.py`**：主要耗时在「每序列全部帧 × 读图与 numpy 统计」（如 20+20 则 40 帧）。粗估：单条约 **0.05～4 s**（SSD 偏低、HDD 偏高，取决于 550×550 PNG 与磁盘）；总时间可近似为 **`(样本数 / NUM_WORKERS) × 单条耗时`**，并加上进程池开销。若 tqdm 显示单条很慢，优先检查磁盘是否为网络盘或 HDD。
- **鲲鹏 920 等多核 CPU**（例如 96 核）：默认 `NUM_WORKERS_CAP=32`，不会启动近百进程；若数据在**本地 NVMe** 且 tqdm 显示磁盘未饱和，可**逐步把 `NUM_WORKERS_CAP` 调到 48～64** 试跑对比；若数据在**网络盘**，有时 **8～16** 进程反而更快。
- 可先对 `data_list.csv` **截取前几百行** 试跑 `03`，用 tqdm 的 **it/s** 推算全量时间。
