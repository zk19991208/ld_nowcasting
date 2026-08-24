SimVP Lightning 封装备份目录（本项目无 git 历史时用于手工保留版本）

1) simvp_lit_full_snapshot.py
   - 与当前 models/simvp.py 内容一致的完整快照（含：冻结 Encoder/Mid/Decoder、验证 CSI 按 batch_idx 间隔、sync_dist 等）。
   - 若日后改坏 simvp.py，可将本文件整体复制覆盖回 models/simvp.py（文件名须为 simvp.py，且仍在 models 包内，相对导入 .simvp_core 才有效）。

2) simvp_lit_no_freeze_for_old_ckpt.py
   - 去掉「参数冻结」与「仅优化可训练参数」的版本；优化器仍为全量 Adam，类名仍为 SimVP_Lit，便于与仅关心旧 ckpt 加载/旧训练脚本的用户对齐。
   - 使用前同样必须复制为 models/simvp.py（覆盖），不要在本子目录下直接 import。

说明：一般情况下，旧 ckpt 与当前 models/simvp.py（freeze 全关）仍可 load_from_checkpoint；若遇兼容问题，再用 2) 覆盖。

train.py 若有 resume_weights_only、freeze_* 等参数，与无冻结版 simvp 并存时仍兼容（未使用的 CLI 参数可忽略）。
