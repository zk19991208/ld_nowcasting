#!/usr/bin/env bash
# SimVP + MovingMNIST（Phys）：在线生成训练序列 + 固定测试序列。
# 等价 YAML 配置见 ../../ascend_run/configs/simvp_mmnist_phys.yaml（可在 ascend_run 下用 run_train_yaml.py 启动）。
# 数据目录需含 train-images-idx3-ubyte.gz 与 mnist_test_seq.npy（见 transfer/data/moving_mnist）。
# 用法：在 Git Bash/WSL 下 chmod +x 后执行；或按路径改成本机 transfer 根目录后运行。

set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/moving_mnist}"

cd "$ROOT"
python -u train.py \
  --model_name simvp \
  --dataset_name MovingMnistDataPhysModule \
  --root_dir "$DATA_ROOT" \
  --height 64 --width 64 \
  --input_length 10 --target_length 10 \
  --input_class 0 --predict_class 0 --predict_class_vmax 1 \
  --loss_fx l2 \
  --learning_rate 0.0001 \
  --batch_size 16 \
  --num_workers 4 \
  --pin_memory 1 \
  --gpus 1 \
  --tensorboard_save_path "$ROOT/log" \
  --tensorboard_exp_name simvp_mmnist_phys \
  --save_dirpath "$ROOT/save/simvp_mmnist" \
  --save_monitor valid_loss_fx \
  --save_filename 'weights-{epoch:03d}-{valid_loss_fx:.3f}' \
  --save_top_k 5 \
  --save_mode min \
  --max_epochs 50 \
  --check_val_rate 0.5 \
  --refresh_rate 20
