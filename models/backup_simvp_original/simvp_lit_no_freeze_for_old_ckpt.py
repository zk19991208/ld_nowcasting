# 本文件为「无参数冻结、优化器含全量参数」的 SimVP_Lit，供与旧训练/旧推理流程完全对齐时
# 将内容覆盖回 models/simvp.py 使用（建议先保留 simvp_lit_full_snapshot.py）。
# 与 simvp_lit_full_snapshot 相比：已删除 freeze_encoder 等及 _apply_parameter_freeze；
# 保留 validation 按 batch_idx 间隔记录 CSI_*_val、sync_dist、on_validation_epoch_end 聚合等修复。
#
# 用法：复制替换
#   copy /Y models\backup_simvp_original\simvp_lit_no_freeze_for_old_ckpt.py models\simvp.py
# （Linux） cp models/backup_simvp_original/simvp_lit_no_freeze_for_old_ckpt.py models/simvp.py

import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loss import get_loss_fx
from metrics import add_score_log, add_time_score_log
from utils import add_parser, add_tensorboard

from .simvp_core import SimVPNet, stride_generator


def _num_spatial_downsample(N_S):
    return sum(1 for s in stride_generator(N_S) if s == 2)


class SimVP_Lit(pl.LightningModule):

    def __init__(self, height, width, input_length, target_length, downscale_factor, learning_rate, loss_fx,
                 input_class, predict_class, predict_class_vmax, add_video, weights_prec, thresholds_prec,
                 weights_radar, thresholds_radar, visual_prec_vmin, visual_prec_vmax, visual_train_steps,
                 visual_val_steps, train_log_steps, val_log_steps, test_save_path, batch_size,
                 hid_S, hid_T, N_S, N_T, incep_ker, groups,
                 lr_scheduler_mode, lr_scheduler_patience, lr_scheduler_factor, lr_scheduler_monitor,
                 lr_scheduler_frequency,
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

        self.loss_fx = get_loss_fx(loss_fx, predict_class, predict_class_vmax, weights_prec, thresholds_prec,
                                    weights_radar, thresholds_radar)
        self._val_epoch_mse_chunks = []
        self._val_epoch_mae_chunks = []

    def on_validation_epoch_start(self):
        self._val_epoch_mse_chunks.clear()
        self._val_epoch_mae_chunks.clear()

    def forward(self, seqs_x):
        return self.net(seqs_x)

    def training_step(self, batch, batch_idx):
        seqs_x, seqs_y = batch
        decoder_frames = self(seqs_x)
        loss = self.loss_fx(decoder_frames, seqs_y)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
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
        return parent_parser
