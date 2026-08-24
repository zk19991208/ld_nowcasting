# 文献与代码阅读路线

## 1. 阅读目标

阅读时重点区分三类“潜表示”：

1. **任务表征**：为预测、控制或分类保留有用信息，不要求重建所有像素。
2. **感知压缩**：通过 Encoder/Decoder 降低空间维度，同时保持足够的视觉或物理信息。
3. **文件压缩**：还需要量化、熵模型和真实码率优化，本项目当前不研究这一方向。

建议带着以下问题阅读：

- 潜表示是单个全局向量，还是二维/时空网格？
- Encoder 是通过重建、未来预测还是遮挡预测训练的？
- 潜空间使用确定性 AE、KL-VAE、VQ-VAE 还是其他正则化？
- Decoder 是否参与主模型训练？
- 下采样比例和潜通道数是多少？
- 论文评价的是视觉质量、物理数值、预报技巧还是控制性能？

## 2. 第一组：理解 LeWM 与 JEPA

### 2.1 LeWorldModel

- 论文：[LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels](https://arxiv.org/abs/2603.19312)
- 官方代码：[lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)
- 项目页：[LeWorldModel](https://le-wm.github.io/)

阅读重点：

```text
ViT-Tiny Encoder
单个 192 维 CLS token
下一潜状态 MSE
SIGReg 防坍缩
Encoder 与 Predictor 端到端联合训练
Decoder 仅用于训练后的可视化
```

需要注意：LeWM 是 reconstruction-free 世界模型，不是 AutoEncoder。它证明潜状态可通过未来可预测性学习，但不能直接证明单 CLS token 足以恢复定量雷达场。

### 2.2 V-JEPA

- 论文与官方代码：[facebookresearch/jepa](https://github.com/facebookresearch/jepa)

阅读重点：

```text
保留时空 patch token 网格
预测被遮挡区域的潜表示
没有正式像素 Decoder
使用冻结 Encoder 后训练的额外 Decoder 做可视化
```

与本项目的关系：V-JEPA 说明 JEPA 并不要求将图像压成单个 token，空间 patch 表征也可以用于潜空间预测。

## 3. 第二组：理解空间 AutoEncoder

### 3.1 Latent Diffusion Models

- 论文：[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- 代码：[CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)

阅读重点：

```text
空间潜特征图 C x H/f x W/f
ResNet CNN Encoder/Decoder
KL regularization 与 VQ regularization
f=4、f=8 的压缩与细节权衡
潜变量尺度对后续生成模型的影响
```

与本项目的关系：LDM 提供了成熟的“CNN 主干 + 低分辨率 Attention + 空间潜特征图”设计，但其自然图像感知损失不能未经验证直接用于 dBZ 场。

### 3.2 Auto-Encoding Variational Bayes

- 论文：[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

阅读重点：

```text
mu 与 logvar
重参数化采样
重建项与 KL 项
VAE 为什么得到连续潜空间
KL 过强时的 posterior collapse
```

## 4. 第三组：雷达潜空间预测

### 4.1 PreDiff

- 论文：[PreDiff: Precipitation Nowcasting with Latent Diffusion Models](https://arxiv.org/abs/2307.10422)
- 会议论文 PDF：[NeurIPS 2023 PDF](https://papers.neurips.cc/paper_files/paper/2023/file/f82ba6a6b981fbbecf5f2ee5de7db39c-Paper-Conference.pdf)
- 官方代码：[gaozhihan/PreDiff](https://github.com/gaozhihan/PreDiff)

这是压缩阶段最优先阅读的雷达先例。

SEVIR 配置：

```text
单帧输入：128 x 128 x 1
空间潜变量：16 x 16 x 4
空间下采样：f=8
逐帧 2D VAE
```

网络组件：

```text
3x3 Conv
ResNet Block
GroupNorm
SiLU
Downsample/Upsample
最低分辨率 Self-Attention
```

需要特别检查：

- VAE 的重建损失、KL 权重和对抗训练设置；
- SEVIR 的数据范围与本项目 dBZ 数据的差异；
- `16x16x4` 潜变量对强降水细节的影响；
- 哪些代码可以只作为架构参考，哪些数据处理不能直接复用。

### 4.2 LDcast

- 论文：[Latent diffusion models for generative precipitation nowcasting with accurate uncertainty quantification](https://arxiv.org/abs/2304.12891)
- 官方代码：[MeteoSwiss/ldcast](https://github.com/MeteoSwiss/ldcast)

阅读重点：

```text
3D CNN VAE
同时压缩时间、高度和宽度
两级下采样后时空网格点减少 64 倍
32 个潜通道
KL 高斯正则化
潜空间概率扩散预报
```

与本项目的关系：适合后续研究时空联合压缩和概率预报，但第一版逐帧压缩不应直接照搬时间维压缩。

### 4.3 PreDiff 与 LDcast 对比

| 项目 | PreDiff | LDcast |
|---|---|---|
| 压缩方式 | 逐帧 2D VAE | 序列 3D VAE |
| 潜表示 | 空间网格 | 时空网格 |
| 时间是否压缩 | 否 | 是 |
| 后续模型 | Earthformer-UNet latent diffusion | Forecaster + denoiser latent diffusion |
| 第一版参考价值 | 高 | 中 |

## 5. 第四组：学习式文件压缩背景

这一组用于理解“降低潜张量维度”与“降低实际码率”的区别，不是第一版实现目标。

### 5.1 End-to-end Optimized Image Compression

- 论文：[End-to-end Optimized Image Compression](https://arxiv.org/abs/1611.01704)

阅读重点：非线性分析/合成变换、量化近似和率失真目标。

### 5.2 Scale Hyperprior

- 论文：[Variational Image Compression with a Scale Hyperprior](https://arxiv.org/abs/1802.01436)

阅读重点：潜变量概率模型、hyperprior、真实码率与重建失真的联合优化。

### 5.3 综述与基准

- [Learning End-to-End Lossy Image Compression: A Benchmark](https://arxiv.org/abs/2002.03711)
- [Learning-Driven Lossy Image Compression: A Comprehensive Survey](https://arxiv.org/abs/2201.09240)

## 6. 雷达临近预报综述

- [Deep Learning for Precipitation Nowcasting: A Survey from the Perspective of Time Series Forecasting](https://arxiv.org/abs/2406.04867)

阅读重点：

```text
雷达数据预处理
递归预测与直接多步预测
损失函数
评价指标
确定性与概率预报
```

这篇综述不是专门讨论潜空间压缩，但有助于确定后续 Predictor 和评价体系。

## 7. 建议阅读顺序

### 最短路线

如果目标是尽快确定第一版：

1. LeWM：方法和训练目标；
2. PreDiff：附录中的 SEVIR VAE 结构表；
3. LDM：感知压缩与 `f=4/f=8` 对比；
4. LDcast：3D VAE 与时空压缩；
5. 回到 `01_compression_plan.md` 确定第一版配置。

### 深入路线

```text
AE/VAE 基础
-> 学习式图像压缩
-> LDM 空间自编码器
-> PreDiff/LDcast 雷达应用
-> LeWM/V-JEPA 可预测表征
-> 本项目的可解码雷达潜空间
```

## 8. 阅读记录模板

每篇论文建议记录：

```text
论文：
输入数据与物理量：
Encoder：
潜变量形状：
Decoder：
下采样比例：
潜空间约束：
重建损失：
是否联合训练 Predictor：
评价指标：
可直接借鉴部分：
不适合本项目部分：
仍有疑问：
```
