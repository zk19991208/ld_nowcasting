# 雷达回波外推短临预报系统

基于 PyTorch Lightning 2.x 构建的雷达回波外推（Radar Echo Extrapolation）训练与评估框架，支持多种深度学习模型和损失函数的灵活组合。输入过去 1 小时（10 帧 × 6 分钟）的雷达反射率，预测未来 2 小时（20 帧 × 6 分钟）的雷达回波演变。

---

## 目录

- [项目结构](#项目结构)
- [环境依赖](#环境依赖)
- [数据说明](#数据说明)
- [快速开始](#快速开始)
- [各模块详细说明](#各模块详细说明)
  - [数据索引构建 — build_index.py](#1-数据索引构建--build_indexpy)
  - [数据加载 — dataset.py](#2-数据加载--datasetpy)
  - [模型架构 — model.py 及模型文件](#3-模型架构--modelpy-及模型文件)
  - [损失函数 — losses.py](#4-损失函数--lossespy)
  - [评估指标 — metrics.py](#5-评估指标--metricspy)
  - [可视化回调 — callbacks.py](#6-可视化回调--callbackspy)
  - [训练入口 — train.py](#7-训练入口--trainpy)
  - [批量实验 — run_experiments.py](#8-批量实验--run_experimentspy)
- [配置文件详解](#配置文件详解)
- [训练](#训练)
- [验证与评估](#验证与评估)
- [模型部署](#模型部署)
- [实验结果参考](#实验结果参考)

---

## 项目结构

```
nowcast_case/
├── build_index.py          # 数据索引构建（一次性运行）
├── dataset.py              # 数据集与 DataModule
├── model.py                # Lightning 模块 + UNet + MEFM
├── model_diffcast.py       # DiffCast 模型（SimVP + 残差扩散）
├── model_wadepre.py        # WADEPre 模型（小波双分支）
├── model_wadepre_v2.py     # WADEPre v2（更贴近原论文实现）
├── model_alphapre.py       # AlphaPre 模型（幅度-相位解耦）
├── losses.py               # 损失函数注册表与组合损失
├── metrics.py              # CSI / POD / FAR 评估指标
├── callbacks.py            # 训练可视化回调
├── train.py                # 训练入口脚本
├── run_experiments.py       # 批量对比实验运行器
├── config.yaml             # 默认配置文件
├── index/                  # 数据索引目录（build_index 生成）
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── meta.json
└── experiments/            # 实验输出目录
    ├── EXP_A/              # 各实验独立目录
    │   ├── config.yaml
    │   ├── checkpoints/
    │   ├── lightning_logs/
    │   ├── vis_output/
    │   ├── train.log
    │   └── results.json
    └── summary.md
```

---

## 环境依赖

```
Python >= 3.10
PyTorch >= 2.0
Lightning >= 2.0
numpy
pyyaml
matplotlib
einops
ptwt          # PyTorch 小波变换（WADEPre 模型依赖）
```

安装示例：

```bash
conda create -n nowcast "python==3.10"

conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install lightning numpy pyyaml matplotlib einops ptwt tensorboard wandb
```

---

## 数据说明

### 数据来源

使用南京大学 C 波段双偏振雷达（NJU-CPOL）观测数据，存储在 `/liyang/data/NJU_CPOL` 目录下。

### 数据组织

```
NJU_CPOL/
├── dBZ/
│   ├── data_dir_001/       # 事件 001
│   │   ├── frame_000.npy   # 单帧雷达反射率，256×256
│   │   ├── frame_001.npy
│   │   └── ...
│   ├── data_dir_002/
│   └── ...                 # 共 258 个事件
├── ZDR/                    # 差分反射率（预留）
└── KDP/                    # 差分传播相移率（预留）
```

- **时间分辨率**：6 分钟/帧
- **空间分辨率**：256 × 256 像素
- **数据格式**：NumPy `.npy` 文件，float32
- **归一化**：框架内部自动将 dBZ 值裁剪到 [0, 70] 并归一化到 [0, 1]

### 变量归一化参数

| 变量 | 最小值 (vmin) | 最大值 (vmax) | 归一化公式 |
|------|-------------|-------------|-----------|
| dBZ  | 0           | 70          | `(x - 0) / 70` |
| ZDR  | -1          | 5           | `(x + 1) / 6` |
| KDP  | -1          | 10          | `(x + 1) / 11` |

---

## 快速开始

### 1. 构建数据索引

首次使用需要生成索引文件，后续训练直接复用：

```bash
cd code
python build_index.py \
  --data_root /liyang/data/NJU_CPOL \
  --output_dir ./index \
  --n_input 10 \
  --n_output 20 \
  --train_ratio 0.8 \
  --val_ratio 0.1
```

输出三个 CSV 文件（`train.csv`, `val.csv`, `test.csv`），每行记录 `(event_id, start_frame)`，以及一个 `meta.json` 记录划分统计信息。

### 2. 训练模型

```bash
# 使用默认配置（UNet + MSE + PM 损失）
python train.py

# 指定配置文件和输出目录
python train.py --config config.yaml --output-dir experiments/my_run

# 命令行覆盖参数
python train.py --train.max_epochs 200 --train.lr 0.0005

# 从 checkpoint 恢复训练
python train.py --config config.yaml --ckpt-path experiments/my_run/checkpoints/epoch05-val_loss0.0023.ckpt
```

### 3. 查看结果

训练完成后自动输出：
- `checkpoints/`：最优的 3 个模型权重
- `lightning_logs/`：TensorBoard 日志
- `vis_output/`：预测对比可视化图
- `results.json`：最终测试指标

```bash
# 启动 TensorBoard 查看训练曲线
tensorboard --logdir experiments/my_run/lightning_logs
```

---

## 各模块详细说明

### 1. 数据索引构建 — `build_index.py`

**功能**：扫描原始雷达数据目录，按事件级别划分训练/验证/测试集，生成滑动窗口样本索引。

**核心逻辑**：
1. 遍历 `data_root/dBZ/data_dir_*` 统计每个事件的帧数
2. 过滤掉帧数不足 `n_input + n_output`（默认 30 帧 = 3 小时）的事件
3. 按事件粒度随机打乱并按比例划分（默认 80/10/10）
4. 对每个事件，以滑动窗口方式枚举所有可用的 `(event_id, start_frame)` 样本

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_root` | 必填 | NJU_CPOL 根目录 |
| `--output_dir` | `./index` | 索引文件输出目录 |
| `--n_input` | 10 | 输入帧数 |
| `--n_output` | 20 | 输出帧数 |
| `--train_ratio` | 0.8 | 训练集事件比例 |
| `--val_ratio` | 0.1 | 验证集事件比例 |
| `--seed` | 42 | 随机种子 |

---

### 2. 数据加载 — `dataset.py`

**功能**：提供 `RadarDataset` 和 `RadarDataModule`，根据索引 CSV 按需加载 `.npy` 帧数据。

**`RadarDataset`**：
- 从 CSV 索引读取 `(event_id, start_frame)`
- 按配置的变量列表（如 `[dBZ]`）加载对应子目录的帧
- 归一化到 [0, 1]
- 返回 `(input, target)` 张量，形状均为 `(T × C, H, W)`

**`RadarDataModule`**（Lightning DataModule）：
- 自动构建 train/val/test 的 DataLoader
- 训练集 shuffle，验证/测试集不 shuffle
- 支持 `persistent_workers` 和 `pin_memory` 加速

**配置（`cfg["data"]`）**：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `data_root` | — | NJU_CPOL 根目录 |
| `index_dir` | `./index` | 索引文件目录 |
| `variables` | `[dBZ]` | 加载变量列表 |
| `n_input_frames` | 10 | 输入帧数 |
| `n_output_frames` | 20 | 输出帧数 |
| `batch_size` | 32 | 每 GPU 批量大小 |
| `num_workers` | 4 | DataLoader 工作进程数 |

---

### 3. 模型架构 — `model.py` 及模型文件

#### Lightning 模块 — `RadarNowcastModule`

**`model.py`** 中的 `RadarNowcastModule` 是整个训练的核心 Lightning 模块，负责：
- 根据配置实例化不同的骨干网络
- 组合损失函数计算
- 管理训练/验证/测试流程
- 计算和记录评估指标
- 配置优化器和学习率调度器

**支持的模型类型**（通过 `cfg["model"]["type"]` 选择）：

#### (a) UNet（默认）

经典 4 层编码器-解码器结构，带跳跃连接：

```
输入 (T_in×C, H, W)
  → Encoder: 4级下采样 [64, 128, 256, 512]
  → Bottleneck: 1024 通道
  → [可选 MEFM 模块]
  → Decoder: 4级上采样 + skip connection
  → Head: 1×1 卷积 → (T_out×C, H, W)
```

**MEFM 模块**（Multi-scale Extraction and Fusion Module，Yang & Yuan 2023）：
- 放置在编码器和解码器之间
- 包含金字塔池化注意力（Pyramid Pooling Attention）和跨尺度注意力（Inter-Scale Attention）
- 通过 `model.mefm.enabled: true` 启用

**配置**：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `model.base_channels` | 64 | 基础通道数（逐层翻倍） |
| `model.mefm.enabled` | false | 是否启用 MEFM |
| `model.mefm.num_heads` | 4 | 注意力头数 |
| `model.mefm.pool_ratios` | [1,2,4,8] | 金字塔池化比率 |

#### (b) WADEPre — `model_wadepre.py`

**论文**：*Wavelet-based Dual-branch Extrapolation for Precipitation Nowcasting*（arXiv:2602.02096）

小波域双分支架构：
1. 对输入做多级小波分解（Haar/db1）
2. **近似分支**：ConvLSTM 处理低频近似系数
3. **细节分支**：卷积网络处理高频细节系数
4. **融合网络（Refiner）**：小波逆变换重构 + 残差精炼

**配置**（`model.wadepre`）：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `spatial_size` | 256 | 输入空间分辨率 |
| `hidden_size` | 128 | ConvLSTM 隐藏层大小 |
| `wavelet_level` | 3 | 小波分解层数 |
| `refine_hidden` | 128 | Refiner 隐藏通道 |

#### (c) WADEPre v2 — `model_wadepre_v2.py`

更忠实于原论文的实现，包含完整的课程学习损失策略：
- 近似分支：多层 ConvLSTM
- 细节分支：IDR（Iterative Detail Refinement）+ FPN
- Refiner：大容量残差网络
- 内置 `compute_wadepre_loss()` 支持论文中的多尺度课程损失

#### (d) AlphaPre — `model_alphapre.py`

**论文**：*AlphaPre: Amplitude-Phase Disentanglement for Precipitation Nowcasting*

频域幅度-相位解耦预测：
1. FFT 将输入分解为幅度谱和相位谱
2. **幅度分支（AmpliNet）**：预测未来的频谱幅度
3. **相位分支（PhaseNet）**：预测未来的频谱相位
4. **频谱融合（AlphaMixer）**：掩膜加权融合 + IFFT 重构

**配置**（`model.alphapre`）：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `spatial_size` | 256 | 输入空间分辨率 |
| `hidden_dim` | 64 | 隐藏层维度 |
| `n_layers` | 3 | 网络层数 |
| `spec_num` | 20 | 频谱分量数 |

#### (e) DiffCast — `model_diffcast.py`

**论文**：*DiffCast: A Unified Framework via Residual Diffusion for Precipitation Nowcasting*（CVPR 2024）

确定性骨干 + 残差扩散框架：
1. **SimVP Backbone**：确定性预测全局运动趋势 μ
2. **ContextNet**：ConvGRU 多尺度时序特征提取，为扩散过程提供条件
3. **GTUNet**：时序 UNet 去噪器，条件化于扩散时间步、片段位置和上下文特征
4. **GaussianDiffusion**：在残差空间 (r = y - μ) 上执行 DDPM 训练 + DDIM 采样

**训练流程**：联合训练 backbone 和扩散模型

```
L_total = α × L_diffusion + (1 - α) × L_backbone
```

**配置**（`model.diffcast`）：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `dim` | 64 | UNet 基础通道 |
| `dim_mults` | [1,2,4,8] | 通道倍增因子 |
| `diffusion_timesteps` | 1000 | 扩散步数 |
| `sampling_timesteps` | 250 | DDIM 采样步数 |
| `objective` | pred_v | 预测目标（v-prediction） |
| `loss_alpha` | 0.5 | 扩散损失权重 |
| `simvp_hid_S` | 64 | SimVP 空间隐藏维度 |
| `simvp_hid_T` | 256 | SimVP 时间隐藏维度 |

> **注意**：DiffCast 显存占用较大，建议使用 `batch_size: 1`（per GPU）+ `bf16-mixed` 精度。

---

### 4. 损失函数 — `losses.py`

**功能**：提供 14 种损失函数的注册表，通过配置文件灵活组合。

**`NowcastLoss` 类**：读取 `cfg["loss"]["components"]`，按权重加权求和：

```
L_total = Σ (weight_i × loss_i(pred, target))
```

**可用损失函数**：

| 注册名 | 说明 | 额外参数 | 来源 |
|--------|------|----------|------|
| `mse` | 均方误差 | — | 标准 |
| `l1` | L1 损失 | — | 标准 |
| `weighted_mse` | 指数加权 MSE | `exponent` (默认 2) | — |
| `weighted_l1` | 指数加权 L1 | `exponent` (默认 2) | — |
| `balanced_mse` | 阈值平衡 MSE | — | Yang & Yuan 2023 |
| `balanced_mae` | 阈值平衡 MAE | — | Yang & Yuan 2023 |
| `pm` | MSE 概率匹配损失 | — | Cao et al. 2025 |
| `pm_l1` | L1 概率匹配损失 | — | Cao et al. 2025 |
| `dice` | Dice 损失 | `threshold` (默认 0.286) | — |
| `ssim` | 结构相似性损失 | — | — |
| `spatial_ms_ssim` | 多尺度空间 SSIM（DWT） | `n_levels`, `weight_*` | Yang & Yuan 2023 |
| `temporal` | 时序一致性损失 | `n_output_frames`, `n_vars` | Yang & Yuan 2023 |

**配置示例**：

```yaml
# 示例 1：MSE + 概率匹配（Cao et al. 2025）
loss:
  components:
    mse: {weight: 1.0}
    pm:  {weight: 10.0}

# 示例 2：CM 损失（Yang & Yuan 2023）
loss:
  components:
    balanced_mse: {weight: 1.0}
    balanced_mae: {weight: 1.0}
    spatial_ms_ssim: {weight: 1.0, n_levels: 3}
    temporal: {weight: 1.0, n_output_frames: 20, n_vars: 1}

# 示例 3：MSE + Dice（强回波优化）
loss:
  components:
    mse: {weight: 1.0}
    dice: {weight: 1.0, threshold: 0.286}
```

---

### 5. 评估指标 — `metrics.py`

**功能**：基于 `torchmetrics.Metric` 实现像素级雷达评估指标。

**`RadarMetrics` 类**：
- 输入：归一化到 [0,1] 的预测和真值张量
- 内部转换为 dBZ 值后按阈值二值化
- 累计 TP（命中）、FP（虚报）、FN（漏报）
- 计算 CSI、POD、FAR

**指标定义**：

| 指标 | 公式 | 含义 |
|------|------|------|
| **CSI** (Critical Success Index) | TP / (TP + FP + FN) | 综合命中率 |
| **POD** (Probability of Detection) | TP / (TP + FN) | 检测概率 |
| **FAR** (False Alarm Ratio) | FP / (TP + FP) | 虚警率 |

**默认阈值**：20 dBZ, 35 dBZ, 40 dBZ

- 20 dBZ ≈ 小雨/弱回波
- 35 dBZ ≈ 中等降水
- 40 dBZ ≈ 强对流

**可选功能**：`per_leadtime: true` 启用逐时刻（Lead Time）评估。

---

### 6. 可视化回调 — `callbacks.py`

**功能**：在训练和验证过程中生成预测 vs 真值的雷达图对比。

**`NowcastVisualizationCallback`**：
- 两行布局：第一行为模型预测，第二行为真值标签
- 每列对应一个预报时刻（通过 `display_lead_minutes` 配置）
- 使用雷达反射率配色方案（0-70 dBZ）
- 输出到 TensorBoard / WandB 和本地 PNG 文件

**配置（`cfg["vis"]`）**：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | true | 是否启用可视化 |
| `display_lead_minutes` | [6,12,...,120] | 显示的预报时刻（分钟） |
| `every_n_epochs` | 1 | 每 N 个 epoch 可视化一次 |
| `train_every_n_steps` | 100 | 训练中每 N 步可视化一次 |
| `val_every_n_steps` | 5 | 验证中每 N 步可视化一次 |
| `save_dir` | `./vis_output` | PNG 保存目录 |

---

### 7. 训练入口 — `train.py`

**功能**：统一的模型训练入口，支持所有模型类型。

**流程**：

```
加载配置 → 构建 DataModule → 实例化模型
  → 配置 Callbacks（Checkpoint、EarlyStopping、可视化）
  → 配置 Loggers（TensorBoard / WandB）
  → trainer.fit() 训练
  → trainer.test() 在最优 checkpoint 上测试
  → 保存 results.json
```

**命令行参数**：

| 参数 | 说明 |
|------|------|
| `--config` | YAML 配置文件路径（默认 `config.yaml`） |
| `--output-dir` | 输出目录（checkpoint, log, vis, results） |
| `--ckpt-path` | 从 checkpoint 恢复训练 |
| `--key.subkey value` | 覆盖配置文件中的任意参数 |

**自动功能**：
- 多 GPU 自动 DDP
- Early Stopping（监控 `val_loss`）
- 保存 Top-3 最优 checkpoint
- 训练结束自动测试并保存指标

---

### 8. 批量实验 — `run_experiments.py`

**功能**：定义并顺序运行多组对比实验，自动汇总结果。

**预定义实验（12 组）**：

| ID | 模型 | 损失函数 | MEFM |
|----|------|----------|------|
| A | UNet | MSE（基线） | ✗ |
| B | UNet | MSE + PM(ω=10) | ✗ |
| C | UNet | Weighted MSE | ✗ |
| D | UNet | Balanced MSE + MAE | ✗ |
| E | UNet | MSE + SSIM | ✗ |
| F | UNet | CM loss（完整） | ✗ |
| G | UNet | MSE + PM + MEFM | ✓ |
| H | UNet | CM loss + MEFM | ✓ |
| I | UNet | MSE + Dice | ✗ |
| J | WADEPre | MSE | ✗ |
| K | AlphaPre | MSE | ✗ |
| L | DiffCast | 扩散联合损失 | ✗ |

**使用方法**：

```bash
# 运行所有实验
python run_experiments.py

# 只运行指定实验
python run_experiments.py --only A,B,I,L

# 仅重建汇总表（不重新训练）
python run_experiments.py --summary-only
```

---

## 配置文件详解

完整的 `config.yaml` 结构：

```yaml
# ── 数据 ──
data:
  data_root: /liyang/data/NJU_CPOL    # 原始数据根目录
  index_dir: ./index                   # 索引文件目录
  variables: [dBZ]                     # 加载变量（可扩展: [dBZ, ZDR, KDP]）
  n_input_frames: 10                   # 输入帧数（过去 1 小时）
  n_output_frames: 20                  # 输出帧数（未来 2 小时）
  batch_size: 32                       # 每 GPU 批量大小
  num_workers: 4                       # DataLoader 工作进程

# ── 模型 ──
model:
  type: unet                           # unet / wadepre / wadepre_v2 / alphapre / diffcast
  base_channels: 64                    # UNet 基础通道数
  mefm:
    enabled: false                     # 是否启用 MEFM 模块
    num_heads: 4
    pool_ratios: [1, 2, 4, 8]
  # wadepre: {...}                     # WADEPre 参数（按需配置）
  # alphapre: {...}                    # AlphaPre 参数（按需配置）
  # diffcast: {...}                    # DiffCast 参数（按需配置）

# ── 训练 ──
train:
  max_epochs: 500                      # 最大训练轮数
  lr: 0.001                            # 初始学习率
  weight_decay: 0.00001                # 权重衰减
  optimizer: adam                      # adam / adamw
  early_stopping_patience: 10          # 早停耐心值
  precision: "16-mixed"                # 训练精度 (16-mixed / bf16-mixed / 32)
  gradient_clip_val: null              # 梯度裁剪值
  scheduler:
    type: plateau                      # plateau / cosine
    patience: 3                        # ReduceLROnPlateau 耐心值
    factor: 0.1                        # 学习率衰减系数
    min_lr: 0.000001                   # 最低学习率

# ── 损失 ──
loss:
  components:
    mse: {weight: 1.0}
    pm:  {weight: 10.0}
  # use_diffcast_loss: true            # DiffCast 专用（启用扩散联合损失）
  # use_wadepre_loss: true             # WADEPre v2 专用（启用课程损失）

# ── 评估 ──
eval:
  thresholds_dbz: [20, 35, 40]        # CSI/POD/FAR 评估阈值
  per_leadtime: false                  # 是否逐时刻评估

# ── 可视化 ──
vis:
  enabled: true
  display_lead_minutes: [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]
  every_n_epochs: 1
  train_every_n_steps: 100
  val_every_n_steps: 5
  save_dir: ./vis_output

# ── 日志 ──
logging:
  use_tensorboard: true
  use_wandb: false
  wandb_project: radar_nowcast
  wandb_entity: null
```

---

## 训练

### 单模型训练

```bash
# 1. 构建索引（首次）
python build_index.py --data_root /liyang/data/NJU_CPOL

# 2. 使用默认 UNet + MSE+PM 训练
python train.py --output-dir experiments/baseline

# 3. 使用 DiffCast 训练（需调整配置）
python train.py \
  --config experiments/DiffCast_full/config.yaml \
  --output-dir experiments/DiffCast_full
```

### 多 GPU 训练

框架自动检测 GPU 数量并使用 DDP（Distributed Data Parallel）：

```bash
# 自动使用所有可用 GPU
python train.py --output-dir experiments/run1

# 指定 GPU 数量
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --output-dir experiments/run1
```

### 从断点恢复

```bash
python train.py \
  --config experiments/run1/config.yaml \
  --output-dir experiments/run1 \
  --ckpt-path experiments/run1/checkpoints/epoch05-val_loss0.0023.ckpt
```

### 后台训练

```bash
nohup python -u train.py \
  --config config.yaml \
  --output-dir experiments/my_run \
  > experiments/my_run/run.log 2>&1 &
```

### 批量对比实验

```bash
# 运行所有 12 组实验
nohup python -u run_experiments.py > experiments/all.log 2>&1 &

# 仅运行 DiffCast 和 Dice 损失实验
python run_experiments.py --only I,L
```

---

## 验证与评估

### 训练过程中的自动评估

每个 epoch 结束时自动计算验证集指标（CSI、POD、FAR），记录到 TensorBoard。

### 测试集评估

训练完成后自动使用最佳 checkpoint 在测试集上评估，结果保存在 `results.json`：

```json
{
  "epochs_trained": 34,
  "best_val_loss": 0.002350,
  "test_metrics": {
    "test_loss": 0.003851,
    "test/CSI_20dBZ": 0.266300,
    "test/POD_20dBZ": 0.289200,
    "test/FAR_20dBZ": 0.191600,
    "test/CSI_35dBZ": 0.060700,
    "test/CSI_40dBZ": 0.019200
  }
}
```

### 手动评估已有 checkpoint

```python
import yaml, lightning as L
from dataset import RadarDataModule
from model import RadarNowcastModule

cfg = yaml.safe_load(open("config.yaml"))
dm = RadarDataModule(cfg)
model = RadarNowcastModule(cfg)

trainer = L.Trainer(accelerator="auto", devices=1)
trainer.test(model, datamodule=dm,
             ckpt_path="experiments/run1/checkpoints/epoch05-val_loss0.0023.ckpt")
```

### TensorBoard 监控

```bash
tensorboard --logdir experiments/ --port 6006
```

可查看：训练/验证损失曲线、CSI/POD/FAR 指标、学习率变化、预测对比图。

---

## 模型部署

### 1. 导出为 TorchScript

```python
import torch, yaml
from model import RadarNowcastModule

cfg = yaml.safe_load(open("config.yaml"))
model = RadarNowcastModule.load_from_checkpoint(
    "experiments/run1/checkpoints/epoch05-val_loss0.0023.ckpt",
    cfg=cfg
)
model.eval()

# 提取骨干网络
net = model.net

# Trace 导出
dummy = torch.randn(1, 10, 256, 256)  # (B, T_in×C, H, W)
traced = torch.jit.trace(net, dummy)
traced.save("model_traced.pt")
```

### 2. 导出为 ONNX

```python
import torch, yaml
from model import RadarNowcastModule

cfg = yaml.safe_load(open("config.yaml"))
model = RadarNowcastModule.load_from_checkpoint(
    "experiments/run1/checkpoints/epoch05-val_loss0.0023.ckpt",
    cfg=cfg
)
model.eval()
net = model.net

dummy = torch.randn(1, 10, 256, 256)
torch.onnx.export(
    net, dummy, "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)
```

### 3. 推理服务示例

```python
import torch
import numpy as np

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("model_traced.pt", map_location=device)
model.eval()

# 准备输入：最近 10 帧（60 分钟）的 dBZ 数据
frames = []
for i in range(10):
    frame = np.load(f"latest_frame_{i:03d}.npy")  # (256, 256)
    frame = np.clip(frame, 0, 70) / 70.0           # 归一化到 [0, 1]
    frames.append(frame)

inp = torch.tensor(np.stack(frames), dtype=torch.float32)  # (10, 256, 256)
inp = inp.unsqueeze(0).to(device)                           # (1, 10, 256, 256)

# 推理
with torch.no_grad():
    pred = model(inp)  # (1, 20, 256, 256)

# 还原为 dBZ
pred_dbz = pred.cpu().numpy()[0] * 70.0  # (20, 256, 256)

# 各时刻预测
for t in range(20):
    minutes = (t + 1) * 6
    print(f"+{minutes}min: max={pred_dbz[t].max():.1f} dBZ")
```

### 4. 部署注意事项

| 项目 | 建议 |
|------|------|
| **UNet 推理速度** | 单帧 ~5ms (GPU) / ~50ms (CPU)，适合实时部署 |
| **DiffCast 推理速度** | 扩散采样约 30-60s/样本，仅建议 GPU 部署；可仅用 backbone（SimVP）做快速推理 |
| **内存占用** | UNet ~260 MB；DiffCast ~700 MB |
| **输入格式** | 归一化到 [0, 1] 的 float32 张量，形状 `(B, T_in×C, H, W)` |
| **输出格式** | [0, 1] 范围的 float32，乘以 70 即为 dBZ 值 |
| **批量推理** | UNet 支持大 batch；DiffCast 受显存限制建议 batch=1 |

---

## 实验结果参考

基于 NJU-CPOL 数据集的对比实验结果（按 CSI_20dBZ 排序）：

| 排名 | 实验 | 模型 | 损失函数 | CSI_20 | CSI_35 | FAR_20 |
|------|------|------|----------|--------|--------|--------|
| 1 | I | UNet | MSE + Dice | **0.3888** | 0.0801 | 0.4797 |
| 2 | D | UNet | Balanced MSE+MAE | **0.3738** | 0.0986 | 0.5037 |
| 3 | F | UNet | CM loss | 0.3425 | 0.0997 | 0.2992 |
| 4 | G | UNet+MEFM | MSE+PM+MEFM | 0.3008 | 0.1004 | 0.4522 |
| 5 | B | UNet | MSE+PM | 0.2990 | 0.0954 | 0.4738 |
| 6 | J | WADEPre | MSE | 0.2715 | 0.0498 | 0.2444 |
| 7 | A | UNet | MSE（基线） | 0.2663 | 0.0607 | 0.1916 |
| 8 | L | DiffCast | 扩散联合损失 | ~0.41* | ~0.13* | ~0.25* |

*DiffCast 验证集指标，测试集结果约 0.25-0.26（仅使用 backbone 推理）。

**结论**：
- Dice 损失在提升弱回波命中率（CSI_20dBZ）方面效果最显著
- CM loss 在 CSI 和 FAR 之间取得了较好的平衡
- MEFM 模块对强回波（CSI_35/40）有一定帮助
- DiffCast 的 backbone（SimVP）在验证集上表现最优，但测试集泛化有待提升

---

## 参考文献

1. Cao et al. (2025). Probability-matching loss for precipitation nowcasting. *GRL*.
2. Yang & Yuan (2023). CM loss and MEFM for radar echo extrapolation. *GRL*.
3. DiffCast (Yu et al., 2024). A unified framework via residual diffusion for precipitation nowcasting. *CVPR*.
4. WADEPre (2024). Wavelet-based dual-branch extrapolation for precipitation nowcasting.
5. AlphaPre (2024). Amplitude-phase disentanglement for precipitation nowcasting.
