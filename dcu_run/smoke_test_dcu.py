# 曙光 DCU 环境烟测：检查 torch 是否识别设备、简单前向+反传，不跑完整训练。
# 运行：在已 module load DTK 并激活含 DTK 版 PyTorch 的 conda 后执行
#   conda activate <你的环境>
#   cd transfer
#   python dcu_run/smoke_test_dcu.py

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError as e:
        print("错误: 无法 import torch:", e, file=sys.stderr)
        return 1

    print("torch:", torch.__version__)
    print("cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device_count:", torch.cuda.device_count())
        print("device 0:", torch.cuda.get_device_name(0))

    try:
        import pytorch_lightning as pl
        print("pytorch_lightning:", pl.__version__)
    except ImportError:
        print("pytorch_lightning: 未安装（训练前请 pip install pytorch-lightning）")

    if not torch.cuda.is_available():
        print("警告: cuda 不可用，DCU 上请确认已加载 DTK 且使用平台提供的 PyTorch。", file=sys.stderr)
        return 2

    device = torch.device("cuda:0")
    x = torch.randn(4, 8, device=device, requires_grad=True)
    w = torch.randn(8, 4, device=device, requires_grad=True)
    y = (x @ w).sum()
    y.backward()
    print("smoke matmul+backward: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
