# 雷达图像压缩阶段设计

## 1. 阶段目标

第一阶段只研究单帧雷达图像的有损潜表示压缩：

```text
X_t -> Encoder -> Z_t -> Decoder -> X'_t
```

这里的“压缩”是为后续潜空间预测降低空间维度，并非生成类似 PNG/JPEG 的二进制码流。当前阶段不加入 ARPredictor、风场条件或多步预测。

输入暂按当前项目配置记为：

```text
X_t: [B, 1, 560, 560]
```

第一阶段需要回答三个问题：

1. 雷达图像在多大的空间下采样比例下仍能可靠重建？
2. 潜表示需要多少通道才能保留弱回波、边界和强对流核心？
3. 适合重建的潜空间是否也具有连续、稳定和可预测的时间演化？

## 2. 与 LeWM 的关系

LeWM 的主训练不是 AutoEncoder 训练。其原始流程为：

```text
o_t -> ViT-Tiny -> z_t
z_t, a_t -> Predictor -> z_hat_{t+1}

L_LeWM = MSE(z_hat_{t+1}, z_{t+1}) + lambda * SIGReg(z)
```

LeWM 将一帧图像压缩为单个 192 维 CLS token，不使用像素重建损失。Decoder 仅在主模型训练完成后作为诊断工具单独训练。

雷达任务必须输出定量反射率场，因此本项目不能完全采用 reconstruction-free 方案。拟借鉴的是：

- 在潜空间预测未来状态；
- 让编码器学习时间上可预测的表示；
- 使用 SIGReg 等方法防止潜空间坍缩或尺度失衡。

需要保留的雷达特有要求是：

- 潜状态保留二维空间网格；
- Decoder 参与正式训练与推理；
- 重建和预测均需要关注 dBZ 数值，而不仅是视觉真实性。

## 3. 推荐的基础架构

第一版候选采用逐帧 2D ResNet AutoEncoder：

```text
Conv 3x3
-> ResBlock x2
-> Downsample x2
-> ResBlock x2
-> Downsample x2
-> ResBlock x2
-> 可选的第三次 Downsample
-> Bottleneck ResBlock x2
-> Latent Projection
```

Decoder 使用对称结构：

```text
Latent Projection
-> Bottleneck ResBlock x2
-> Upsample + Conv
-> ResBlock x2
-> Upsample + Conv
-> ResBlock x2
-> 可选的第三次 Upsample
-> Conv 3x3
```

基础组件建议：

```text
卷积：3x3 Conv2d
激活：SiLU
归一化：GroupNorm
下采样：stride=2 的卷积
上采样：nearest/bilinear interpolation + 3x3 Conv
残差：仅使用 ResBlock 内部短残差
```

明确不使用跨越瓶颈的 U-Net 长跳跃连接。否则 Decoder 可能依赖 Encoder 的高分辨率特征，导致潜状态 `Z_t` 没有承载完整重建信息；预测未来时也不存在可用的未来跳连特征。

## 4. 空间潜表示候选

### 4.1 f=4：保真度基线

```text
X_t: [B, 1, 560, 560]
Z_t: [B, C, 140, 140]
```

当 `C=8` 时：

```text
原始元素数：1 * 560 * 560 = 313600
潜变量元素数：8 * 140 * 140 = 156800
元素压缩约 2 倍
```

### 4.2 f=8：主要候选

```text
X_t: [B, 1, 560, 560]
Z_t: [B, C, 70, 70]
```

当 `C=8` 时：

```text
原始元素数：1 * 560 * 560 = 313600
潜变量元素数：8 * 70 * 70 = 39200
元素压缩约 8 倍
```

### 4.3 潜通道数

候选值为：

```text
C in {4, 8, 16}
```

不应只报告空间下采样倍数。实际压缩率需要同时考虑潜通道数：

```text
compression_ratio = (1 * H * W) / (C * h * w)
```

## 5. CNN 与 Attention 候选

### 5.1 纯 CNN 基线

优先实现纯 ResNet CNN AutoEncoder。其优势是局地归纳偏置明确、训练稳定、对数据量要求较低，并且适合重建回波边界和小尺度强中心。

### 5.2 瓶颈局部 Attention

纯 CNN 基线稳定后，可在最低分辨率加入：

```text
Window Attention
Swin Block
Axial Attention
```

对于 `70x70=4900` 个空间位置，不建议直接使用全局自注意力。局部窗口或轴向注意力更节省显存。

### 5.3 纯 ViT

纯 ViT 不是第一版候选。它可以保留完整 patch token 网格，但从头训练更依赖数据量，全局注意力成本较高，且大 patch 可能损失小尺度强回波。后续可将其作为单独的架构对比，而不是基础实现。

## 6. 潜空间约束候选

Encoder/Decoder 主体保持一致，通过输出头和损失切换不同潜空间约束。

### 6.1 普通 AE

```text
Z = Encoder(X)
L = L_reconstruction
```

用途：建立最大重建保真度基线。

### 6.2 弱 KL-VAE

```text
mu, logvar = Encoder(X)
Z = mu + exp(0.5 * logvar) * epsilon

L = L_reconstruction + lambda_kl * L_KL
```

用途：检验连续、规则的潜空间能否改善后续预测。`lambda_kl` 应从较小值开始，避免强回波细节被过度平滑。

### 6.3 AE + SIGReg

```text
Z = Encoder(X)
L = L_reconstruction + lambda_sigreg * L_SIGReg
```

用途：在保持确定性编码的同时，借鉴 LeWM 约束潜表示的总体分布。SIGReg 如何应用于空间潜特征图仍需专门设计和验证，不能直接假定所有空间位置是独立样本。

## 7. 重建损失候选

第一版建议从数值保真损失开始：

```text
L_reconstruction = L_weighted_L1
                 + alpha * L_MSE
                 + beta * L_gradient
```

候选强回波权重：

```text
dBZ < 20: weight=1
20 <= dBZ < 35: weight=2
35 <= dBZ < 45: weight=4
dBZ >= 45: weight=8
```

实际阈值和权重必须结合数据归一化方式实现，损失内部判断应基于可解释的物理量或与之严格对应的归一化值。

第一版暂不加入：

- GAN 判别损失；
- 面向自然图像的感知损失；
- 复杂频域损失；
- 时间一致性损失。

这些损失应在数值重建基线完成后分别消融，避免无法判断改进来自何处。

## 8. 第一轮最小实验矩阵

先比较压缩率，再比较潜空间约束，避免一次改变过多变量。

### 8.1 压缩率实验

| 实验 | 架构 | 下采样 | 潜通道 | 潜空间约束 |
|---|---|---:|---:|---|
| C1 | ResNet AE | 4 | 8 | 无 |
| C2 | ResNet AE | 8 | 8 | 无 |
| C3 | ResNet AE | 8 | 4 | 无 |
| C4 | ResNet AE | 8 | 16 | 无 |

### 8.2 正则化实验

在压缩率实验选出的结构上比较：

| 实验 | 潜空间方法 | 目的 |
|---|---|---|
| R1 | 普通 AE | 重建基线 |
| R2 | 弱 KL-VAE | 检查连续潜空间 |
| R3 | AE + SIGReg | 检查 LeWM 式分布约束 |

### 8.3 架构实验

最后比较：

| 实验 | Bottleneck | 目的 |
|---|---|---|
| A1 | ResBlock | CNN 基线 |
| A2 | ResBlock + Window Attention | 检查大尺度组织结构 |

## 9. 评价指标

### 9.1 全场数值指标

```text
MAE
MSE
RMSE
```

### 9.2 阈值指标

```text
CSI@25/35/45 dBZ
BIAS@25/35/45 dBZ
POD@25/35/45 dBZ
FAR@25/35/45 dBZ
```

### 9.3 强回波专项指标

```text
强回波区域 MAE/RMSE
强回波面积误差
最大 dBZ 偏差
连通区域数量与面积变化
```

### 9.4 可视化检查

每次实验至少固定输出相同个例的：

```text
原始图
重建图
绝对误差图
25/35/45 dBZ 阈值轮廓
强回波局部放大图
```

## 10. 第一版选择规则

第一版架构应在文献阅读后确定，建议遵循以下决策规则：

1. 先淘汰强回波 CSI、BIAS 或最大值偏差明显不可接受的压缩设置。
2. 如果 `f=4` 与 `f=8` 重建能力接近，优先选择计算量更低的 `f=8`。
3. 如果增加潜通道只改善全场 MSE，却不改善强回波指标，不增加通道。
4. Attention 只有在大尺度结构指标和典型个例上稳定改善时才保留。
5. AE、VAE 和 SIGReg 的最终选择不能只看重建；还需要在下一阶段比较潜状态的时间平滑性与预测难度。

## 11. 暂缓事项

以下内容不属于压缩阶段第一版：

- 时间维压缩或 3D VAE；
- 潜空间 ARPredictor；
- 风场、地形、ERA5 或 NWP 条件；
- 端到端多步预报微调；
- 概率扩散预测；
- 文件码率、量化和熵编码。

## 12. 当前实现与运行方法

第一版已实现为项目内自包含代码，没有引入 Diffusers、PreDiff、GAN、KL 或 Attention 依赖。

主要文件：

```text
models/radar_autoencoder.py
loss/radar_reconstruction.py
tests/test_radar_autoencoder.py
dcu_run/configs/radar_ae_f4_c8_xinjiang.yaml
dcu_run/configs/radar_ae_f8_c8_xinjiang.yaml
```

模型包含：

```text
ResBlock：GroupNorm + SiLU + 3x3 Conv
Downsample：stride=2 的 3x3 Conv
Upsample：nearest interpolation + 3x3 Conv
Encoder/Decoder：对称层级，无 U-Net 长跳连
输出：二维连续潜特征图
```

为避免把一个 `B x 40 x 1 x 560 x 560` 序列全部展开导致显存过高，训练模块默认从每个样本的输入与目标序列中随机抽取一帧：

```yaml
ae_frame_source: all
ae_frames_per_sample: 1
```

验证阶段使用固定、均匀的帧位置，保证重复验证具有可比性。需要增加单次 batch 中使用的帧数时，可提高 `ae_frames_per_sample`，但应同步降低 `batch_size`。

### 12.1 f=8 主实验

```powershell
python dcu_run/run_train_yaml.py --config dcu_run/configs/radar_ae_f8_c8_xinjiang.yaml
```

默认结构：

```text
base_channels=32
channel_multipliers=[1,2,4,4]
latent_channels=8
560x560 -> 8x70x70
```

### 12.2 f=4 保真度对照

```powershell
python dcu_run/run_train_yaml.py --config dcu_run/configs/radar_ae_f4_c8_xinjiang.yaml
```

默认结构：

```text
base_channels=32
channel_multipliers=[1,2,4]
latent_channels=8
560x560 -> 8x140x140
```

### 12.3 当前损失

```text
L = weighted_L1 + 0.2 * MSE + 0.1 * gradient_L1
```

加权区间默认按目标雷达值划分：

```text
<20 dBZ: 1
20-35 dBZ: 2
35-45 dBZ: 4
>=45 dBZ: 8
```

### 12.4 验证输出

训练过程记录：

```text
valid_loss_fx
val/weighted_l1
val/mse
val/gradient
CSI_25dBZ_val / BIAS_25dBZ_val / POD_25dBZ_val / FAR_25dBZ_val
CSI_35dBZ_val / BIAS_35dBZ_val / POD_35dBZ_val / FAR_35dBZ_val
CSI_45dBZ_val / BIAS_45dBZ_val / POD_45dBZ_val / FAR_45dBZ_val
```

TensorBoard 每轮验证还会固定记录第一批样本的重建图、目标图和绝对误差图。

单元测试覆盖 `f=4/f=8` 的潜变量形状、重建尺寸、反向传播、非法输入尺寸和重建损失基本行为。
