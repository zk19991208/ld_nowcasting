# 雷达潜空间世界模型实验

## 当前进度

- 阶段 1 已完成：采用 `f=4, C=8` 的逐帧 ResNet AutoEncoder；
- 已完成单帧个例推理，结果位于
  `inference/radar_ae_f4_manual/summary.json`；
- 当前进入阶段 2：冻结 AE，训练雷达潜空间 Predictor；
- 阶段 2 的可执行实验设计见
  [03_predictor_plan.md](./03_predictor_plan.md)。

## 文档导航与当前状态

- [压缩阶段设计与实验矩阵](./01_compression_plan.md)：整理第一阶段的模型候选、训练方法、评价指标和决策规则。
- [文献与代码阅读路线](./02_literature.md)：整理 LeWM、PreDiff、LDcast、LDM 和 V-JEPA 等参考资料。
- [Predictor 训练设计](./03_predictor_plan.md)：整理第二阶段的冻结策略、模型、损失、基线和实验顺序。

当前已经确定的原则：

1. 最终潜表示保留二维空间结构，不采用 LeWM 的单个 CLS token。
2. Decoder 是雷达模型的正式组成部分，因为最终输出必须是定量反射率场。
3. 第一阶段先解决逐帧图像压缩与重建，不在这一阶段混入时间预测。
4. 不使用跨越瓶颈的 U-Net 长跳跃连接，避免 Decoder 绕过潜表示。
5. 先建立简单、可解释的 CNN 基线，再根据实验决定是否加入 VAE、SIGReg 或局部 Attention。

当前已经确定的第一版选择：

```text
AutoEncoder：确定性 ResNet AE，f=4，C=8
潜状态形状：8 x 140 x 140
Predictor：潜空间 SimVP 式直接 20 -> 20 多步预测
训练方式：冻结 AE，先 latent Huber，再小权重 decoded field loss
```

当前代码进度：阶段 1 的确定性 ResNet CNN AutoEncoder 和单帧推理已经完成；阶段 2
的直接多步 Predictor、latent 统计脚本和 P1 配置已经实现，运行方法见
[Predictor 训练设计](./03_predictor_plan.md#14-当前实现与运行方法)。

## 研究动机

本实验探索一种面向雷达回波外推的潜空间预测框架。其核心思想是不直接在原始高维图像空间中预测未来雷达场，而是先将雷达帧压缩为紧凑的潜状态，在潜空间中学习时间演化规律，再将预测得到的潜状态解码为雷达反射率场。

该设计借鉴了 LeWM 等潜空间世界模型，但最终目标有所不同：雷达外推必须输出定量的物理场，而不仅是用于规划的未来潜状态。

## 整体框架

```text
雷达序列 X_{t-k:t}
        |
        v
雷达自编码器的编码器
        |
        v
潜状态序列 Z_{t-k:t}
        |
        +----------------------+
        |                      |
        v                      v
LeWM 风格的 ARPredictor    条件编码器
        |                 预报时效 / 风场 / 外部强迫
        +----------+-----------+
                   |
                   v
预测潜状态序列 Z_hat_{t+1:t+n}
                   |
                   v
雷达自编码器的解码器
                   |
                   v
预测雷达序列 X_hat_{t+1:t+n}
```

## 模型模块

### 1. 雷达自编码器：编码器

训练面向雷达数据的编码器，将每一帧雷达图像压缩为潜表示：

```text
X_t -> Encoder -> Z_t
```

编码器应保留雷达回波的主要空间形态，包括回波边界、局地纹理、对流核心和弱回波区域。可尝试基于 CNN 的编码器，或基于图像块的 ViT/Swin 风格编码器。对于雷达外推，保留空间特征图可能比将整帧压缩为单个 CLS token 更有效。

### 2. 雷达自编码器：解码器

解码器从潜状态重建雷达反射率场：

```text
Z_t -> Decoder -> X'_t
```

在 LeWM 中，解码器主要用于可视化；而在本实验中，解码器是雷达预测模型的正式组成部分，因为模型最终必须输出具有物理意义的反射率场。

推荐的重建损失为：

```text
L_AE = L1(X'_t, X_t) + alpha * MSE(X'_t, X_t) + beta * L_strong_echo
```

其中，`L_strong_echo` 可对反射率超过 25、35 或 45 dBZ 的像素赋予更高权重。

### 3. LeWM 风格的自回归预测器

借鉴 LeWM 的自回归潜状态预测思想：

```text
Z_{t-k:t}, C_{t+1:t+n} -> ARPredictor -> Z_hat_{t+1:t+n}
```

这里的 `C` 不再表示机器人动作，而表示气象条件或外部强迫。预测器学习的是潜空间动力学，而不是图像空间中的演变过程。

候选输入包括：

```text
历史潜状态 Z_history
预报时效嵌入 lead-time embedding
时刻 / 年积日嵌入
风场潜表示
地形 / 海陆掩膜
NWP 背景场或 ERA5 强迫变量
```

### 4. 条件编码器

可将原始 LeWM 中的动作编码器重新解释为条件编码器。

对于标量或类别条件：

```text
预报时效 -> embedding / MLP -> 条件向量
小时、月份、季节 -> embedding -> 条件向量
```

对于网格化气象强迫场：

```text
风场 / 温度 / 湿度场 -> CNN 编码器 -> 条件潜表示
```

可选的条件来源包括：

```text
预报时效
U/V 风场
垂直风切变
地形高度
ERA5 或 NWP 背景场
边界条件
```

### 5. 高斯潜空间正则化

LeWM 本身不是 AutoEncoder：它不依赖像素重建训练编码器，而是联合训练 Encoder 与潜状态 Predictor，并使用 SIGReg 防止表征坍缩。本实验只借鉴其“可预测潜空间”和高斯分布正则化思想；是否在雷达 AutoEncoder 中采用 SIGReg，需要通过消融实验确定。

正则项应使潜表示保持足够的信息量和分散性：

```text
Z ~ N(0, I)
```

概念性目标函数为：

```text
L_reg = distribution_regularization(Z)
```

该正则项需要谨慎使用。正则过强可能损害重建质量，正则过弱则可能产生缺乏结构、难以预测的潜空间。

## 训练策略

### 阶段 1：训练雷达自编码器

使用单帧雷达数据，或将雷达序列逐帧输入，训练编码器和解码器：

```text
X_t -> Encoder -> Z_t -> Decoder -> X'_t
```

第一版基线目标函数：

```text
L = L_AE
```

重建基线稳定后，再分别测试弱 KL-VAE 和 SIGReg：

```text
L_VAE = L_AE + lambda_kl * L_KL
L_SIGReg = L_AE + lambda_sigreg * L_SIGReg
```

重点检查：

```text
重建 MAE / RMSE
25、35、45 dBZ 阈值下的 CSI 和 BIAS
强回波重建质量
潜表示压缩率
X_t 与 X'_t 的可视化对比
```

### 阶段 2：冻结自编码器，训练潜状态预测器

将完整雷达序列编码为潜状态：

```text
X_{t-k:t+n} -> Encoder -> Z_{t-k:t+n}
```

训练 ARPredictor：

```text
Z_{t-k:t}, C_{t+1:t+n} -> Z_hat_{t+1:t+n}
```

目标函数：

```text
L_latent = MSE(Z_hat_{t+1:t+n}, stopgrad(Z_{t+1:t+n}))
```

也可以将预测的潜状态解码，并加入物理场损失：

```text
X_hat_{t+1:t+n} = Decoder(Z_hat_{t+1:t+n})

L_field = L1(X_hat_{t+1:t+n}, X_{t+1:t+n})
```

组合目标函数：

```text
L = L_latent + gamma * L_field
```

### 阶段 3：端到端微调

待自编码器和预测器训练稳定后，使用较小的学习率对完整模型进行端到端微调：

```text
Encoder + Predictor + Decoder
```

目标函数：

```text
L = L_field + lambda_latent * L_latent + lambda_reg * L_reg
```

该阶段为可选项。如果端到端微调破坏了潜空间结构或降低了重建质量，则继续冻结自编码器。

## 建议实验

### 实验 A：自编码器压缩实验

目标：验证雷达帧能否被有效压缩并重建。

对比内容：

```text
不同的潜空间通道数
不同的下采样比例
CNN 编码器与图像块编码器
使用 / 不使用强回波加权损失
使用 / 不使用高斯潜空间正则化
```

### 实验 B：无条件潜状态预测

目标：检验仅依靠潜空间动力学能否实现雷达外推。

```text
Z_history -> ARPredictor -> Z_future
```

### 实验 C：预报时效条件

目标：检验显式引入预报时效嵌入能否改善多步预测。

```text
Z_history + lead_time -> ARPredictor -> Z_future
```

### 实验 D：风场 / 外部强迫条件

目标：检验气象动力条件能否改善回波的移动和强度演变预测。

```text
Z_history + wind_latent -> ARPredictor -> Z_future
```

### 实验 E：与图像空间模型对比

将潜空间框架与现有图像空间模型进行对比：

```text
SimVP
ConvLSTM
PhyDNet
Earthformer
```

评估指标：

```text
MSE / MAE / RMSE
25、35、45 dBZ 阈值下的 CSI
25、35、45 dBZ 阈值下的 BIAS
分预报时效评分
典型个例可视化
```

## 关键问题

1. 雷达自编码器在压缩后能否保留强回波结构？
2. 在预测能力明显下降之前，可接受的潜表示压缩率是多少？
3. 高斯潜空间正则化能否提高潜状态预测的稳定性？
4. CLS token 风格的全局潜表示是否足够，还是必须保留空间潜特征图？
5. 风场或外部强迫条件能否改善回波位移和增强过程的预测？
6. 与图像空间预测相比，潜空间预测能否降低计算成本？

## 工作假设

雷达回波外推可以表述为一个气象潜空间世界模型：

```text
压缩后的雷达状态 + 气象条件 -> 未来的压缩雷达状态
```

其中，自编码器负责空间压缩，ARPredictor 学习潜空间中的时间演化，条件编码器注入物理强迫信息，解码器则将预测的潜状态映射回定量雷达反射率场。
