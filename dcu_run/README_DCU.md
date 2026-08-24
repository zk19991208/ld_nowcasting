# 曙光 DCU 上运行 transfer 训练

本项目训练入口为根目录下的 `train.py`，通过 PyTorch Lightning 的 `accelerator` 选择设备。**DCU 节点应使用 `accelerator: gpu`**（DTK 提供的 PyTorch 通常通过 CUDA 兼容接口暴露设备），**不要**使用昇腾专用的 `npu`（见 `ascend_run/README_ASCEND.md`）。

## 环境准备（在 DCU 计算节点上）

1. **加载 DTK / 编译器栈**（版本以机房为准，示例）  
   ```bash
   module load compiler/dtk/23.10
   ```
   具体 `module` 名称与版本请咨询集群文档或管理员。

2. **PyTorch**  
   曙光 DCU 基于 ROCm 路线，**勿**直接使用 PyTorch 官网的 CUDA 轮子；应使用平台提供的 **DTK 配套 PyTorch whl**（路径随集群变化，常见在软件仓 `DeepLearning/whl/dtk-*/pytorch/` 下）。

3. **Python 依赖**  
   ```bash
   pip install pyyaml pytorch-lightning tensorboard imageio numpy
   ```

4. **自检**  
   ```bash
   python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.__version__)"
   hy-smi
   ```
   也可运行本目录 `smoke_test_dcu.py`。

## 启动训练

在 `transfer/dcu_run` 下（或把 `--config` 写成绝对路径）：

```bash
python run_train_yaml.py --config configs/simvp_xinjiang_cr_550_original_png_dcu.yaml
python run_train_yaml.py --config configs/simvp_xinjiang_cr_550_original_png_dcu.yaml --dry_run
```

修改 YAML 中的 `radar_dir`、`train_file` 等为 DCU 节点上的实际路径；`devices` 与申请的 DCU 数量一致。`--devices` 会覆盖 `gpus`，与 ascend 配置习惯相同。

### 指定其中几块 DCU（例如 8 卡只用后 6 块：物理编号 2–7）

物理编号一般为 `0,1,...,7`。后 6 块即 **`2,3,4,5,6,7`**。

**方法一（推荐）：环境变量重映射**

进程里只看到 6 块卡，编号变成 `0..5`，再用数量启动：

```bash
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
python run_train_yaml.py --config configs/your.yaml
```

YAML 里写 **`devices: 6`**，且不要与下面「指定序号」混用。

**方法二：只写 `gpus`，不要写 `devices`**

```yaml
accelerator: gpu
gpus: "2,3,4,5,6,7"
# 不要写 devices，否则整数 devices 会覆盖 gpus（见 train.py）
```

若写了 `devices: 2`，含义是「用 2 张卡」，不是「用 2 号卡」。

## 多卡与通信

默认 `train.py` 使用 `strategy="ddp"`。若多卡报错，可参考机房说明设置 **RCCL/NCCL 相关环境变量**（如网卡绑定等），此处不硬编码版本相关参数。

## 与昇腾的区别（速查）

| 项目       | 昇腾 (ascend_run) | 曙光 DCU (dcu_run) |
|------------|-------------------|---------------------|
| accelerator | `npu`             | `gpu`               |
| PyTorch    | torch_npu + CANN | DTK 配套 torch（常为 HIP/CUDA API） |
| 进程策略   | 文档可能写 ddp_npu | 一般标准 `ddp` 即可 |
