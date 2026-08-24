# SimVP 的 PyTorch Lightning 封装，与 transfer 中 PhyDNet_ATT 等训练流程对齐（同一 DataModule、loss、指标与 TB）。
# 约束：input_length 须等于 target_length；len(input_class) 须等于 len(predict_class)；H、W 须能被 Encoder 下采样整除。
# 运行：在项目 transfer 目录下执行
#   python train.py --model_name simvp ...（hid_S/hid_T/N_S/N_T 等）
# 载入 finetune 权重并只冻 Encoder（Mid/Decoder 继续训练）：加 --freeze_encoder 1 --weight_path .../xxx.ckpt

import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loss import get_loss_fx
from loss.frequency_loss import BasePlusMultiScaleFrequency
from metrics import add_score_log, add_time_score_log
from utils import add_parser, add_tensorboard

from .simvp_core import SimVPNet, stride_generator


def _num_spatial_downsample(N_S):
    return sum(1 for s in stride_generator(N_S) if s == 2)


def _freeze_flag(v) -> bool:
    """兼容 argparse int(0/1) 与 YAML bool。"""
    if isinstance(v, bool):
        return v
    return bool(int(v))


def _arg_zero_one(s) -> int:
    """命令行 / YAML 经 run_train_yaml 可能为 0、1 或 true、false 字符串。"""
    if isinstance(s, bool):
        return int(s)
    sl = str(s).lower().strip()
    if sl in ("true", "yes", "1"):
        return 1
    if sl in ("false", "no", "0", ""):
        return 0
    i = int(sl)
    if i not in (0, 1):
        raise ValueError(f"freeze 开关须为 0/1，当前: {s!r}")
    return i


class SimVP_Lit(pl.LightningModule):

    def __init__(self, height, width, input_length, target_length, downscale_factor, learning_rate, loss_fx,
                 input_class, predict_class, predict_class_vmax, add_video, weights_prec, thresholds_prec,
                 weights_radar, thresholds_radar, visual_prec_vmin, visual_prec_vmax, visual_train_steps,
                 visual_val_steps, train_log_steps, val_log_steps, test_save_path, batch_size,
                 hid_S, hid_T, N_S, N_T, incep_ker, groups,
                 lr_scheduler_mode, lr_scheduler_patience, lr_scheduler_factor, lr_scheduler_monitor,
                 lr_scheduler_frequency,
                 freeze_encoder=0, freeze_mid=0, freeze_decoder=0,
                 freq_loss_weight=0.0, freq_loss_alpha=1.0, freq_loss_scales=None,
                 **kwargs):
        super(SimVP_Lit, self).__init__()
        self.save_hyperparameters()

        assert input_length == target_length, (
            "SimVP 当前实现要求 input_length == target_length，与参考仓库中 T_in=T_out 一致。"
        )
        assert len(input_class) == len(predict_class), (
            "SimVP 编解码同通道数，要求 len(input_class) == len(predict_class)。"
        )

        n_down = _num_spatial_downsample(N_S)
        factor = 2 ** n_down
        assert height % factor == 0 and width % factor == 0, (
            f"height/width 须能被 2^{n_down}={factor} 整除（与 N_S={N_S} 的下采样一致），当前 height={height}, width={width}。"
        )

        self.num_channels = len(predict_class)
        shape_in = (input_length, self.num_channels, height, width)
        self.net = SimVPNet(shape_in, hid_S=hid_S, hid_T=hid_T, N_S=N_S, N_T=N_T,
                            incep_ker=list(incep_ker), groups=groups)

        base_loss = get_loss_fx(loss_fx, predict_class, predict_class_vmax, weights_prec, thresholds_prec,
                                weights_radar, thresholds_radar)
        fw = float(freq_loss_weight)
        if fw > 0.0:
            scales = freq_loss_scales if freq_loss_scales is not None else [1, 2, 4]
            self.loss_fx = BasePlusMultiScaleFrequency(
                base_loss, freq_weight=fw, alpha=float(freq_loss_alpha), scales=list(scales),
            )
        else:
            self.loss_fx = base_loss
        self._val_epoch_mse_chunks = []
        self._val_epoch_mae_chunks = []

        self._apply_parameter_freeze()

    def _apply_parameter_freeze(self):
        """按子模块设置 requires_grad：SimVPNet.enc / .hid / .dec。"""
        fe = _freeze_flag(getattr(self.hparams, "freeze_encoder", 0))
        fm = _freeze_flag(getattr(self.hparams, "freeze_mid", 0))
        fd = _freeze_flag(getattr(self.hparams, "freeze_decoder", 0))
        for p in self.net.enc.parameters():
            p.requires_grad = not fe
        for p in self.net.hid.parameters():
            p.requires_grad = not fm
        for p in self.net.dec.parameters():
            p.requires_grad = not fd
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        if n_train == 0:
            raise ValueError(
                "SimVP_Lit：freeze_encoder/freeze_mid/freeze_decoder 不能同时为真（无可训练参数）。"
            )
        print(
            f"SimVP_Lit 参数冻结: enc={fe}, mid={fm}, dec={fd}；可训练 {n_train:,} / 总计 {n_total:,}",
            flush=True,
        )

    def on_validation_epoch_start(self):
        self._val_epoch_mse_chunks.clear()
        self._val_epoch_mae_chunks.clear()

    def forward(self, seqs_x):
        return self.net(seqs_x)

    def training_step(self, batch, batch_idx):
        seqs_x, seqs_y = batch
        decoder_frames = self(seqs_x)
        loss = self.loss_fx(decoder_frames, seqs_y)

        # 每 N 个 global_step 记一次 add_score_log；须用「% N == 0」。
        # 若写成 if step % N，则 val_log_steps/train_log_steps 为 1 时恒为 0（假），永远不记分项指标。
        _nts = max(100, int(self.hparams.train_log_steps))
        if self.global_step % _nts == 0:
            metrics_pred = add_score_log(self.hparams.predict_class, self.hparams.predict_class_vmax,
                                         seqs_y, decoder_frames, marker="train")
        else:
            metrics_pred = {}

        if self.hparams.add_video and (self.global_step % self.hparams.visual_train_steps == 0):
            add_tensorboard(self.logger, self.hparams.input_class, self.hparams.predict_class,
                            self.hparams.predict_class_vmax,
                            seqs_x, decoder_frames, seqs_y, self.hparams.visual_prec_vmin,
                            self.hparams.visual_prec_vmax,
                            self.global_step, marker="train")
        self.log_dict(
            {**metrics_pred, "train_loss": loss},
            on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seqs_x, seqs_y = batch
        batch_size = seqs_x.shape[0]
        decoder_frames = self(seqs_x)
        decoder_frames = torch.clip(decoder_frames, 0, 1)
        mse_per_frame = torch.zeros(self.hparams.target_length, device=self.device)
        mae_per_frame = torch.zeros(self.hparams.target_length, device=self.device)
        for i in range(self.hparams.target_length):
            mse_per_frame[i] = torch.sum(torch.square(decoder_frames[:, i, ...] - seqs_y[:, i, ...])) / batch_size
            mae_per_frame[i] = torch.sum(torch.abs(decoder_frames[:, i, ...] - seqs_y[:, i, ...])) / batch_size

        mse_total_frame = torch.sum(torch.mean(torch.square(decoder_frames - seqs_y), dim=(0, 1)))
        mae_total_frame = torch.sum(torch.mean(torch.abs(decoder_frames - seqs_y), dim=(0, 1)))
        val_loss = mse_total_frame + mae_total_frame
        valid_loss_fx = self.loss_fx(decoder_frames, seqs_y)

        if self.hparams.add_video and (batch_idx % self.hparams.visual_val_steps == 0):
            add_tensorboard(self.logger, self.hparams.input_class, self.hparams.predict_class,
                            self.hparams.predict_class_vmax,
                            seqs_x, decoder_frames, seqs_y, self.hparams.visual_prec_vmin,
                            self.hparams.visual_prec_vmax,
                            self.global_step, marker="val")
        if batch_idx == 0:
            self.logger.experiment.add_scalars("val/mae_per_frame", {"frame_%02d" % i: mae_per_frame[i] for i in
                                                                     range(self.hparams.target_length)},
                                               self.global_step)
            self.logger.experiment.add_scalars("val/mse_per_frame", {"frame_%02d" % i: mse_per_frame[i] for i in
                                                                     range(self.hparams.target_length)},
                                               self.global_step)
        # 验证时 global_step 不变，不能用 global_step 节流。按本地 batch_idx 每 val_log_steps 跳变记录 CSI 等，
        # batch_idx==0 必记，避免 ModelCheckpoint(monitor='CSI_*_val') 整轮缺键。
        _vs = max(1, int(self.hparams.val_log_steps))
        if batch_idx % _vs == 0:
            metrics_pred = add_score_log(self.hparams.predict_class, self.hparams.predict_class_vmax,
                                         seqs_y, decoder_frames, marker="val")
        else:
            metrics_pred = {}
        metrics_pred = {**metrics_pred, "val_loss": val_loss, "val_mse_total": mse_total_frame,
                        "val_mae_total": mae_total_frame, 'valid_loss_fx': valid_loss_fx.item()}
        self.log_dict(metrics_pred, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._val_epoch_mse_chunks.append(mse_total_frame.detach())
        self._val_epoch_mae_chunks.append(mae_total_frame.detach())

    def on_validation_epoch_end(self):
        # PyTorch Lightning 2.x 已移除 validation_epoch_end，改用本 hook + 在 step 内累积
        if not self._val_epoch_mse_chunks:
            return
        epoch_mse_val = torch.stack(self._val_epoch_mse_chunks).mean()
        epoch_mae_val = torch.stack(self._val_epoch_mae_chunks).mean()
        self.log_dict(
            {"epoch_mse_val": epoch_mse_val, "epoch_mae_val": epoch_mae_val},
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        seqs_x, seqs_y = batch
        batch_size = seqs_x.shape[0]
        decoder_frames = self(seqs_x)
        decoder_frames = torch.clip(decoder_frames, 0, 1)
        if self.hparams.test_save_path:
            for ibatch in range(batch_size):
                dir_test = os.path.join(self.hparams.test_save_path,
                                        "%04d" % (batch_idx * self.hparams.batch_size + ibatch))
                if not os.path.exists(dir_test):
                    os.makedirs(dir_test)
                path_test = os.path.join(dir_test, "pred.npy")
                with open(path_test, "wb") as fp:
                    np.save(fp, decoder_frames.detach().cpu().numpy()[ibatch, ...])

        metrics_pred = add_score_log(self.hparams.predict_class, self.hparams.predict_class_vmax,
                                     seqs_y, decoder_frames, marker="test")
        time_part_pred = add_time_score_log(self.hparams.predict_class, self.hparams.predict_class_vmax,
                                            seqs_y, decoder_frames, marker="test")
        self.log_dict({**metrics_pred, **time_part_pred}, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("SimVP_Lit：无可训练参数，请检查 freeze_* 开关。")
        optimizer = torch.optim.Adam(params, lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode=self.hparams.lr_scheduler_mode,
                                      patience=self.hparams.lr_scheduler_patience,
                                      factor=self.hparams.lr_scheduler_factor)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler,
                                                         'monitor': self.hparams.lr_scheduler_monitor,
                                                         'frequency': self.hparams.lr_scheduler_frequency}}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SimVP_Lit")
        add_parser(parser)
        parser.add_argument('--hid_S', type=int, default=64)
        parser.add_argument('--hid_T', type=int, default=256)
        parser.add_argument('--N_S', type=int, default=4)
        parser.add_argument('--N_T', type=int, default=8)
        parser.add_argument('--groups', type=int, default=8)
        parser.add_argument('--incep_ker', nargs='+', type=int, default=[3, 5, 7, 11])
        parser.add_argument('--lr_scheduler_mode', type=str, default="min")
        parser.add_argument('--lr_scheduler_patience', type=int, default=2)
        parser.add_argument('--lr_scheduler_factor', type=float, default=0.3)
        parser.add_argument('--lr_scheduler_monitor', type=str, default='val_loss')
        parser.add_argument('--lr_scheduler_frequency', type=int, default=1)
        parser.add_argument(
            '--freeze_encoder',
            type=_arg_zero_one,
            default=0,
            help='1 时冻结 SimVPNet Encoder（requires_grad=False），常用于载入预训练后只训 Mid+Decoder。',
        )
        parser.add_argument(
            '--freeze_mid',
            type=_arg_zero_one,
            default=0,
            help='1 时冻结 Mid_Xnet。',
        )
        parser.add_argument(
            '--freeze_decoder',
            type=_arg_zero_one,
            default=0,
            help='1 时冻结 Decoder。',
        )
        return parent_parser
