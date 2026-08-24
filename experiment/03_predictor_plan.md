# 阶段 2：雷达潜空间 Predictor 训练设计

## 1. 阶段目标

冻结已经训练完成的 `f=4, C=8` 确定性 AutoEncoder，只训练时序
Predictor：

```text
X_1:20 -> frozen Encoder -> Z_1:20
Z_1:20 -> Predictor -> Z_hat_21:40
Z_hat_21:40 -> frozen Decoder -> X_hat_21:40
```

当前数据时间间隔为 6 min，因此 20 帧输入和 20 帧输出分别表示过去
120 min 和未来 120 min。阶段 2 的主要问题不是继续提高单帧压缩质量，而是验证
`8 x 140 x 140` 的空间潜状态能否学习稳定的时间演化。

本阶段使用的 AE checkpoint：

```text
/root/private_data/ld_pred/save/radar_ae_f4_c8_xinjiang/
weights-epoch=013-valid_loss_fx=0.00298.ckpt
```

## 2. 阶段 1 结果及其含义

个例结果来自：

```text
inference/radar_ae_f4_manual/summary.json
```

当前文件只有 1 个序列的 6 个手选帧，不能代替完整验证集统计，但可以作为阶段
2 的初步依据：

| 指标 | 6 帧均值 |
|---|---:|
| MAE | 0.095 dBZ |
| RMSE | 0.472 dBZ |
| 最大值绝对误差 | 2.56 dBZ |
| CSI 25 dBZ | 0.734 |
| CSI 35 dBZ | 0.502 |
| BIAS 35 dBZ | 1.61 |

结论：AE 的连续值重建误差较低，可以开始 Predictor；但是强回波区域存在明显的
阈值敏感性和面积偏差。Predictor 的最终评分必须与 AE 自身的重建上限分开，否则
无法判断误差来自时序预测还是 Decoder。

开始正式训练前，应在**完整验证集的全部未来帧**上计算一次：

```text
X_future -> Encoder -> Decoder -> X_oracle
```

`X_oracle` 的指标定义了当前 AE/Decoder 能达到的上限。此过程仍是逐帧重建，不改变
AE 的训练任务。

## 3. 第一版 Predictor 架构

### 3.1 选择：潜空间 SimVP 式直接多步预测

第一版采用全卷积、直接输出 20 个未来潜状态的 Predictor，而不是全局
Transformer 或逐步扩散模型。实现上复用项目已有的 SimVP 时空翻译模块，但输入和
输出都是 AE latent，不直接处理 560 x 560 雷达图像。

```text
输入             B x 20 x 8 x 140 x 140
逐帧空间编码     8 -> hid_S，140 -> 35
时序翻译         将 20 个历史潜状态联合映射为 20 个未来状态
逐帧空间解码     hid_S -> 8，35 -> 140
输出增量         B x 20 x 8 x 140 x 140
```

第一版建议参数：

```yaml
predictor_type: latent_simvp
predictor_hid_s: 32
predictor_hid_t: 256
predictor_n_s: 4
predictor_n_t: 4
predictor_groups: 8
predictor_residual: none
```

第一版直接预测完整的未来潜状态：

```text
Z_hat_21:40 = Predictor(Z_1:20)
```

不再额外执行 `Z_20 + DeltaZ`。SimVP 内部已经包含多尺度编码/解码路径和跳接；额外
加入 last-frame residual 会引入较强的 persistence 先验，可能使模型倾向于复制最后
一帧。模型一次直接预测全部 20 帧，不进行 20 次自回归滚动，因此第一版也不会引入
teacher forcing 和逐步误差累积两个额外变量。

### 3.2 为什么第一版不用全局 Transformer

每帧 latent 有 `140 x 140 = 19600` 个空间位置，20 帧共有 392000 个位置。直接对
全部位置做全局自注意力，显存和计算量都不合适。后续若卷积基线不足，可在
`35 x 35` 的中间特征上加入窗口注意力或 Earthformer，而不是在原始 latent 网格上
使用全局 attention。

### 3.3 为什么第一版不用 ConvLSTM 自回归

ConvLSTM/ConvGRU 是重要对照，但自回归版本同时引入滚动误差、训练/推理分布差异和
scheduled sampling。第一版先使用直接多步模型验证 latent 是否可预测；随后再将
ConvGRU 作为结构对照，而不是一开始混入所有变量。

## 4. AE 的冻结方式

训练阶段必须满足：

```python
encoder.eval()
decoder.eval()
encoder.requires_grad_(False)
decoder.requires_grad_(False)
```

编码历史帧和未来真值时使用 `torch.no_grad()`，不保存 AE Encoder 的反向图。只有在
计算图像场损失时，允许梯度从冻结 Decoder 的输出传回 `Z_hat`；Decoder 参数仍不
更新。

BatchNorm 不应更新。本 AE 使用 GroupNorm，但仍固定为 `eval()`，保证推理一致性。

## 5. 潜状态标准化

当前是确定性 AE，没有 KL 或 SIGReg，8 个 latent 通道的均值、方差和数值尺度可能
不同。直接计算 raw latent MSE 会让高方差通道支配训练。

在训练集上用冻结 Encoder 统计每通道：

```text
mu_c    = mean(Z[:, c, :, :])
sigma_c = std(Z[:, c, :, :])
Z_norm  = (Z - mu_c) / max(sigma_c, 1e-6)
```

将 `mu`、`sigma` 保存在 Predictor checkpoint 中。Predictor 输入、输出和 latent
损失都在标准化空间中；送入 AE Decoder 前执行反标准化：

```text
Z_hat = Z_hat_norm * sigma + mu
```

统计量只允许由训练集生成，不能使用验证集或测试集。第一版按通道统计，不做逐像素
空间统计，避免学习固定地理位置的分布并占用过多参数。

## 6. 训练损失

### 6.1 P1：先用标准化 latent Huber loss 收敛

不直接使用 raw latent MSE，第一版采用对异常值更稳健的 Smooth L1/Huber：

```text
L_lat = mean_j Huber(Z_hat_norm_j, stopgrad(Z_norm_j))
```

20 个未来时效先等权，避免先验上偏向短时效。所有时效必须分别记录验证误差。

### 6.2 P2：加入少量解码场损失微调

latent loss 收敛后，从同一个 checkpoint 继续训练，并在每个 batch 的 20 个未来时效
中随机抽取 2--4 帧解码：

```text
X_hat_j = frozen Decoder(denorm(Z_hat_norm_j))
L_field = RadarReconstructionLoss(X_hat_j, X_j)
L = L_lat + lambda_field * L_field
```

建议从以下参数开始：

```yaml
predictor_field_frames_per_sample: 2
predictor_lambda_field: 0.1
```

这里复用阶段 1 的高值加权 L1、MSE 和梯度损失。只解码少量未来帧是为了控制
`560 x 560` Decoder 反向传播的显存。验证时仍应分块解码全部 20 帧。

暂不加入 GAN、扩散、KL 或 SIGReg。它们不能回答第一版最核心的“确定性 latent 是否
可预测”问题，还会改变优化目标。

### 6.3 暂缓的附加损失

只有在 P2 显示明确问题后再分别消融：

- 35/45 dBZ Soft-CSI 或 Tversky：解决强回波漏报/虚警；
- latent 时间差分损失：约束增长、衰减和移动的连续性；
- lead-time 加权：改变远期与近期预报侧重；
- 频谱或梯度损失：减少空间结构过度平滑。

不要在第一轮同时加入这些损失，否则无法判断收益来源。

## 7. 数据流与显存设计

数据仍使用现有 `SingleRadarDataModule`：

```text
seqs_x: B x 20 x 1 x 560 x 560
seqs_y: B x 20 x 1 x 560 x 560
```

40 帧应按小块送入冻结 Encoder，例如每次编码 4--8 帧，然后恢复为：

```text
z_x: B x 20 x 8 x 140 x 140
z_y: B x 20 x 8 x 140 x 140
```

初始配置建议：

```yaml
batch_size: 1
predictor_encode_chunk_size: 4
precision: 16-mixed
learning_rate: 0.0002
weight_decay: 0.0001
max_epochs: 100
gradient_clip_val: 1.0
```

如果显存足够，优先把 `batch_size` 提到 2；如果不足，先降低 `hid_T` 到 128，而不是
改变 AE 的 f=4 或 latent channel。梯度累积可以提高有效 batch，但不会减少单次前向
所需的峰值显存。

### 7.1 是否预计算 latent

第一版先在线编码，以保证流程简单且不会因重叠序列重复保存大量 latent。记录每个
epoch 的 Encoder 用时占比；只有当冻结 AE 编码明显成为主要瓶颈时，再实现按唯一
雷达帧路径缓存的 float16 latent。不要按 CSV 行缓存整个 40 帧序列，因为滑动窗口会
产生大量重复数据。

## 8. 基线与公平比较

必须同时计算以下四条基线：

| 名称 | 定义 | 作用 |
|---|---|---|
| Raw persistence | 未来 20 帧均复制最后一张输入图 | 最基本外推基线 |
| Latent persistence | `D(E(X_20))` 重复 20 次 | 与 Predictor 使用相同 Decoder |
| AE oracle | `D(E(X_future))` | AE/Decoder 的可达到上限 |
| Latent Predictor | `D(P(E(X_history)))` | 阶段 2 模型 |

Predictor 至少应稳定优于 latent persistence；它与 AE oracle 的差距才是纯时序预测
误差。与 raw persistence 比较可以反映整套潜空间方法是否真正有价值。

## 9. 验证指标

所有最终业务指标在解码后的 dBZ 场上计算，并按预报时效分别记录：

```text
lead = 1, 5, 10, 15, 20
time = 6, 30, 60, 90, 120 min
```

指标包括：

- MAE、RMSE；
- 25/35/45 dBZ 的 CSI、POD、FAR、BIAS；
- 每帧最大 dBZ 偏差；
- 35/45 dBZ 强回波面积偏差；
- 所有 20 个时效的均值和逐时效曲线；
- 标准化 latent Huber/MSE。

当前原始数据为 `550 x 550`，模型输入通过右侧和下侧补零到 `560 x 560`。业务指标
必须先裁回左上角 `550 x 550` 有效区域再累计；补零边界只用于满足网络尺寸要求，
不能参与评分。`summary.json` 中的现有 MAE/RMSE 是初步个例结果，正式基线应按此口径
重新计算。

CSI 必须先在整个验证集累计 hits、misses、false alarms，再计算比值，不能先计算每帧
CSI 后求均值；否则无强回波帧会造成大量空值和统计偏差。

checkpoint 选择建议以稳定的 `val/decoded_weighted_loss` 最小为主，同时额外保存
`val/CSI_35_mean` 最大的 checkpoint。45 dBZ 事件更稀少，第一版不单独用它做 early
stopping monitor。

## 10. 第一轮实验矩阵

一次只改变一个因素：

| 编号 | Predictor | 损失 | 目的 |
|---|---|---|---|
| P0 | Persistence | 无训练 | 建立最低基线与 AE 上限 |
| P1 | Latent SimVP direct | latent Huber | 验证 latent 可预测性 |
| P2 | P1 checkpoint | latent Huber + 0.1 field | 改善解码场与强回波 |
| P3 | Latent SimVP residual | latent Huber | 检查 last-frame residual 是否有益 |
| P4 | Latent ConvGRU | latent Huber | 比较直接多步与自回归结构 |

执行顺序为 `P0 -> P1 -> P2`。只有 P1 明确优于 persistence 后才做 P2；P3/P4 属于
第一版跑通后的结构消融。

## 11. 典型个例输出

固定使用同一组验证样本，至少覆盖：

- 平移为主的稳定回波；
- 强回波增长；
- 强回波消散；
- 分裂或合并；
- 45 dBZ 以上强对流核心；
- 大面积弱回波和接近空场。

每个样本输出 4 行时间序列图：

```text
历史观测 / 未来真值 / AE oracle / Predictor
```

并在 6、30、60、90、120 min 另外输出绝对误差图和 35/45 dBZ 阈值轮廓。个例选择
应预先固定，不能只展示效果好的样本。

## 12. 第一版通过标准

阶段 2 第一版可以继续推进的最低条件：

1. 训练和验证 latent loss 稳定下降，无 NaN、latent 尺度爆炸；
2. 在 6--60 min 的 CSI 25/35 dBZ 上稳定优于 latent persistence；
3. BIAS 没有因高值加权持续明显膨胀；
4. 随 lead time 增长，指标退化连续且可解释；
5. Predictor 与 AE oracle 的差距可被清楚量化；
6. 固定个例中能看到回波移动或演变，而不是简单复制最后一帧。

如果 P1 连短时效都不能超过 persistence，应先检查潜状态标准化、时间顺序、直接
输出定义和数据配对，不应立即加入 GAN 或扩散模型。

## 13. 计划实现文件

下一步编码建议保持独立，避免继续混用从原项目复制的外推代码：

```text
models/radar_latent_predictor.py
loss/radar_predictor.py
inference/infer_radar_latent_predictor.py
dcu_run/configs/radar_predictor_f4_c8_xinjiang.yaml
tests/test_radar_latent_predictor.py
```

`train.py` 只增加新的 `model_name: radar_latent_predictor` 分支；数据读取继续复用
`SingleRadarDataModule`。阶段 1 的 AE 代码和 checkpoint 不做修改。

## 14. 当前实现与运行方法

第一版直接预测代码已经实现：

```text
models/radar_latent_predictor.py
preprocess/compute_radar_ae_latent_stats.py
dcu_run/configs/radar_predictor_f4_c8_xinjiang.yaml
tests/test_radar_latent_predictor.py
```

### 14.1 生成训练集 latent 统计量

在项目根目录运行，只能传训练集 CSV：

```bash
python preprocess/compute_radar_ae_latent_stats.py \
  --ae_checkpoint "save/radar_ae_f4_c8_xinjiang/weights-epoch=013-valid_loss_fx=0.00298.ckpt" \
  --train_file preprocess/xinjiang_cr_6min/data_list_train.csv \
  --radar_dir data/xinjiang/CR_6min_550x550 \
  --output artifacts/radar_ae_f4_c8_xinjiang_latent_stats.npz \
  --input_length 20 \
  --target_length 20 \
  --height 560 \
  --width 560 \
  --encode_chunk_size 8 \
  --device cuda:0
```

首次调试可以加 `--max_batches 10` 检查流程，但正式训练必须删除该参数并扫描完整
训练集。输出文件包含每通道 `mean/std`；Predictor checkpoint 还会把数值复制到自身
hparams 中，后续加载 Predictor 时不依赖原统计文件。

### 14.2 启动 P1

确认 YAML 中 AE checkpoint、数据路径和统计文件路径正确，然后运行：

```bash
python dcu_run/run_train_yaml.py \
  --config dcu_run/configs/radar_predictor_f4_c8_xinjiang.yaml
```

当前 P1 配置明确使用：

```yaml
predictor_residual: none
predictor_lambda_field: 0.0
```

即 SimVP 直接输出未来 20 帧标准化 latent，只用 Huber loss 训练。验证阶段会将全部
未来 latent 分块解码，并记录 `valid_loss_fx`、decoded field loss，以及
25/35/45 dBZ 的整体和指定时效指标。

### 14.3 从 P1 进入 P2

P1 明确优于 persistence 后，复制一份配置并修改：

```yaml
predictor_lambda_field: 0.1
predictor_field_frames_per_sample: 2
resume_path: /path/to/p1/best.ckpt
```

P2 仍冻结 AE，只允许场损失的梯度经过 Decoder 回到预测 latent。不要把 AE checkpoint
填到 `resume_path`；`predictor_ae_checkpoint` 才是阶段 1 AE 权重，`resume_path` 必须是
阶段 2 Predictor checkpoint。
