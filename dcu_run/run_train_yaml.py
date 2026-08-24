# 从 YAML 读取训练参数，在 ld_pred 根目录下调用 train.py（供曙光 DCU 使用）。
# 用法: 在 ld_pred 根目录下执行
#   python run_train_yaml.py --config configs/simvp_xinjiang_cr_550_original_png_dcu.yaml
#   python run_train_yaml.py --config configs/... --dry_run
# 依赖: pip install pyyaml

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from argparse import ArgumentParser
from typing import Any

# 仅由启动器消费，不传给 train.py
_LAUNCHER_KEYS = frozenset({"launcher", "nproc_per_node", "nnodes"})


def _to_cli_args(cfg: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for k, v in cfg.items():
        if k in _LAUNCHER_KEYS:
            continue
        if v is None:
            continue
        flag = f"--{k}"
        if isinstance(v, bool):
            out.extend([flag, "true" if v else "false"])
        elif isinstance(v, list):
            out.append(flag)
            for item in v:
                out.append(str(item))
        else:
            out.extend([flag, str(v)])
    return out


def main() -> None:
    parser = ArgumentParser(description="从 YAML 调用 transfer/train.py（DCU）")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置路径")
    parser.add_argument("--dry_run", action="store_true", help="只打印命令不执行")
    args = parser.parse_args()

    try:
        import yaml
    except ImportError as e:
        raise SystemExit("请先安装 PyYAML: pip install pyyaml") from e

    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML 根节点必须为字典")

    cli_args = _to_cli_args(cfg)

    launcher = str(cfg.get("launcher", "python")).lower()
    if launcher == "torchrun":
        nproc = int(cfg.get("nproc_per_node", 1))
        nnodes = int(cfg.get("nnodes", 1))
        cmd = [
            "torchrun",
            f"--nnodes={nnodes}",
            f"--nproc_per_node={nproc}",
            "train.py",
        ] + cli_args
    else:
        cmd = [sys.executable, "-u", "train.py"] + cli_args

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    printable = " ".join(shlex.quote(x) for x in cmd)
    print("cwd:", project_root)
    print("将执行命令:")
    print(printable)

    if args.dry_run:
        return

    subprocess.check_call(cmd, cwd=project_root)


if __name__ == "__main__":
    main()
