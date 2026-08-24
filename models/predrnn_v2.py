import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loss import get_loss_fx
from metrics import add_score_log
from utils import add_parser, add_tensorboard
from utils.schedule_sampling import reserve_schedule_sampling_exp, schedule_sampling


class PredRNN_v2(LightningModule):

    def __init__(self, height, width, input_length, target_length, downscale_factor, learning_rate, loss_fx,
                 input_class, predict_class, predict_class_vmax, add_video, weights_prec, thresholds_prec,
                 weights_radar, thresholds_radar, visual_prec_vmin, visual_prec_vmax, visual_train_steps,
                 visual_val_steps, train_log_steps, val_log_steps, test_save_path, num_hidden, filter_size, stride,
                 layer_norm, lr_scheduler, lr_scheduler_mode, lr_scheduler_patience, lr_scheduler_factor,
                 lr_scheduler_monitor, lr_scheduler_frequency, reverse_scheduled_sampling, r_sampling_step_1,
                 r_sampling_step_2, r_exp_alpha, scheduled_sampling, sampling_stop_iter, sampling_changing_rate,
                 param_a, param_b, param_c, param_d, share_params, **kwargs):
        """
        :param height: int，输入原始图像的宽度，指裁剪前的图像的宽度
        :param width: int，输入原始图像的宽度，指裁剪前的图像的宽度
        :param input_length: int，输入数据序列的长度
        :param target_length: int，输出数据序列的长度
        :param downscale_factor: int，将输入图像裁剪成patch，该参数代表裁剪的块数， 数量为downscale_factor**2
        :param learning_rate: float，模型的学习率
        :param loss_fx: 损失函数
        :param input_class:  int[list]，0代表降水， 1代表雷达，[0, 1]代表第一个通道放降水，第二个通道放雷达, [0,]代表只放降水
        :param predict_class:  int[list]，0代表降水， 1代表雷达，[0, 1]代表第一个通道放降水，第二个通道放雷达, [0,]代表只放降水
        :param predict_class_vmax: float[list], 对应predict class的最小最大反归一化的最大值
        :param add_video: bool, 是否在tensorboard中输出图像
        :param weights_prec: float[list], 对应降水阈值的权重
        :param thresholds_prec: float[list], 降水分级别阈值, units: mm/6min
        :param weights_radar: float[list], 对应雷达阈值的权重
        :param thresholds_radar: float[list], 雷达反射率分级别阈值, units: dBZ
        :param visual_prec_vmin: float, 降水在tensorboard可视化的vmin
        :param visual_prec_vmax: float, 降水在tensorboard可视化的vmax
        :param visual_train_steps: int, tensorboard在train阶段保存图像的间隔
        :param visual_val_steps: int, tensorboard在val阶段保存图像的间隔
        :param train_log_steps: int, train step的记录间隔
        :param val_log_steps: int, val记录的间隔
        :param test_save_path: str, 测试时文件保存路径
        :param num_hidden: type[list], 列表的长度代表隐藏层的数量，里面的值代表了每一层的隐藏层数量
        :param filter_size: int, 卷积核的大小
        :param stride: int, 卷积核的步长
        :param layer_norm: bool，if layer_norm == 1则采用layer_norm, 默认为0
        :param lr_scheduler: bool, 是否采用 learning scheduler
        :param lr_scheduler_mode:
        :param lr_scheduler_patience:
        :param lr_scheduler_factor:
        :param lr_scheduler_monitor:
        :param lr_scheduler_frequency:
        :param reverse_scheduled_sampling: bool，是否采用reverse_scheduled_sampling的策略
        :param r_sampling_step_1: float，reverse_scheduled_sampling第一阶段的步数
        :param r_sampling_step_2: float，reverse_scheduled_sampling第二阶段的步数
        :param r_exp_alpha: float，reverse_scheduled_sampling的衰减参数
        :param scheduled_sampling: bool，是否采用scheduled_sampling的策略
        :param sampling_stop_iter: int，scheduled_sampling停止的步数
        :param sampling_changing_rate: scheduled_sampling的衰减参数
        :param param_a: float, loss函数的input_loss的加权
        :param param_b: float, loss函数的target_loss的加权
        :param param_c: float, loss函数的解耦损失的加权encoder
        :param param_d: float, loss函数的解耦损失的加权decoder
        :param share_params: bool, encoder和decoder是否share参数
        :param kwargs:
        """
        super(PredRNN_v2, self).__init__()
        self.save_hyperparameters()
        self.hparams.input_length = 20
        self.hparams.target_length = 20
        self.loss_fx = get_loss_fx(loss_fx, predict_class, predict_class_vmax, weights_prec, thresholds_prec,
                                   weights_radar, thresholds_radar)
        self.num_layers = len(num_hidden)
        adapter_num_hidden = num_hidden[0]
        self.num_channels_in = len(input_class)
        self.num_channels_out = len(predict_class)

        self.pixel_unshuffle = nn.PixelUnshuffle(self.hparams.downscale_factor)
        self.pixel_shuffle = nn.PixelShuffle(self.hparams.downscale_factor)

        self.channels_in = self.num_channels_in * self.hparams.downscale_factor ** 2
        self.channels_out = self.num_channels_out * self.hparams.downscale_factor ** 2

        patched_height = height // downscale_factor
        patched_width = width // downscale_factor

        if share_params:
            assert self.num_channels_in == self.num_channels_out, "channels in should equal to channels out!"
            self.encoder = PredRNN_encoder_v2(num_hidden, patched_height, patched_width, filter_size, stride,
                                              layer_norm,
                                              adapter_num_hidden, self.num_layers, self.channels_in)
            self.decoder = self.encoder
        else:
            self.encoder = PredRNN_encoder_v2(num_hidden, patched_height, patched_width, filter_size, stride,
                                              layer_norm,
                                              adapter_num_hidden, self.num_layers, self.channels_in)

            self.decoder = PredRNN_encoder_v2(num_hidden, patched_height, patched_width, filter_size, stride,
                                              layer_norm,
                                              adapter_num_hidden, self.num_layers, self.channels_out)


    def forward(self, frames_tensor_input, mask_input, frames_tensor_output, mask_output, **kwargs):
        """
        ///forward过程
        :param frames_tensor_input: tensor[batch, input_length, channel, height, width]，模型的输入
        :param mask_input: tensor[batch, input_length-1, channel, height, width]，scheduled_sampling用于input阶段的mask输入
        :param frames_tensor_output: tensor[batch, target_length, channel, height, width]，模型的理想输出（仅用于训练，测试和预测设置为0）
        :param mask_output: tensor[batch, target_length-1, channel, height, width]，scheduled_sampling用于output阶段的mask输入
        :param kwargs:
        :return:
        """

        frames_tensor_output = self.pixel_unshuffle(frames_tensor_output)
        frames_tensor_input = self.pixel_unshuffle(frames_tensor_input)

        batch, _, _, height, width = frames_tensor_input.shape
        encoder_frames = []
        decoder_frames = []
        h_t = []
        c_t = []
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.hparams.num_hidden[i], height, width], device=self.device)
            h_t.append(zeros)
            c_t.append(zeros)
        memory = torch.zeros([batch, self.hparams.num_hidden[0], height, width], device=self.device)
        decouple_loss_encoder_sum = 0
        for t in range(self.hparams.input_length):
            if t == 0:
                input_encoder = frames_tensor_input[:, t]
            else:
                encoder_frames.append(x_gen_encoder)
                input_encoder = mask_input[:, t - 1] * frames_tensor_input[:, t] + (
                        1 - mask_input[:, t - 1]) * x_gen_encoder
            h_t, c_t, memory, x_gen_encoder, decouple_loss_encoder = self.encoder(input_encoder, t == 0, h_t, c_t,
                                                                                  memory)
            decouple_loss_encoder_sum += decouple_loss_encoder

        x_gan_decoder = x_gen_encoder[:, :self.channels_out]
        decoder_frames.append(x_gan_decoder)
        decouple_loss_decoder_sum = 0
        for t in range(self.hparams.target_length - 1):
            input_decoder = mask_output[:, t] * frames_tensor_output[:, t] + (1 - mask_output[:, t]) * x_gan_decoder
            _, _, _, x_gen_decoder, decouple_loss_decoder = self.decoder(input_decoder, t == 0, h_t, c_t, memory)

            decoder_frames.append(x_gen_decoder)
            decouple_loss_decoder_sum += decouple_loss_decoder

        encoder_frames = torch.stack(encoder_frames, dim=1)
        decoder_frames = torch.stack(decoder_frames, dim=1)

        encoder_frames = self.pixel_shuffle(encoder_frames)
        decoder_frames = self.pixel_shuffle(decoder_frames)

        return encoder_frames, decoder_frames, decouple_loss_encoder_sum, decouple_loss_decoder_sum

    def training_step(self, batch, batch_idx):
        seqs_x, seqs_y = batch
        batch_size = seqs_x.shape[0]

        if self.hparams.reverse_scheduled_sampling == 1:
            input_flag, target_flag, r_eta, eta = reserve_schedule_sampling_exp(self.global_step,
                                                                                self.hparams.r_sampling_step_1,
                                                                                self.hparams.r_sampling_step_2,
                                                                                self.hparams.r_exp_alpha, batch_size,
                                                                                self.hparams.input_length,
                                                                                self.hparams.target_length, self.device)
        elif self.hparams.scheduled_sampling == 1:
            input_flag, target_flag, r_eta, eta = schedule_sampling(self.global_step, self.hparams.sampling_stop_iter,
                                                                    self.hparams.sampling_changing_rate, batch_size,
                                                                    self.hparams.input_length,
                                                                    self.hparams.target_length,
                                                                    self.device)
        else:
            input_flag = torch.ones((1, self.hparams.input_length - 1, 1, 1, 1), device=self.device)
            target_flag = torch.zeros((1, self.hparams.target_length - 1, 1, 1, 1), device=self.device)
            r_eta = 1
            eta = 0

        encoder_frames, decoder_frames, decouple_loss_encoder, decouple_loss_decoder = self(seqs_x, input_flag, seqs_y,
                                                                                            target_flag)

        input_loss = self.loss_fx(encoder_frames, seqs_x[:, 1:, ...])
        target_loss = self.loss_fx(decoder_frames, seqs_y)

        loss = self.hparams.param_a * input_loss + \
               self.hparams.param_b * target_loss + \
               self.hparams.param_c * decouple_loss_encoder + \
               self.hparams.param_d * decouple_loss_decoder

        if self.global_step % self.hparams.train_log_steps:
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
            {**metrics_pred, "train_loss": loss, 'train_input_loss': input_loss, "train_target_loss": target_loss,
             "train_decouple_loss_encoder": decouple_loss_encoder, "train_decouple_loss_decoder": decouple_loss_decoder,
             "train_r_eta": r_eta, "train_eta": eta, }, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seqs_x, seqs_y = batch
        input_flag = torch.ones(1, self.hparams.input_length - 1, 1, 1, 1, device=self.device)
        target_flag = torch.zeros(1, self.hparams.target_length - 1, 1, 1, 1, device=self.device)

        mse_per_frame = torch.zeros(self.hparams.target_length, device=self.device)
        mae_per_frame = torch.zeros(self.hparams.target_length, device=self.device)

        batch_size = seqs_y.shape[0]

        _, decoder_frames, _, _ = self(seqs_x, input_flag, seqs_y, target_flag)
        decoder_frames = torch.clip(decoder_frames, 0, 1)
        for i in range(self.hparams.target_length):
            mse_per_frame[i] = torch.sum(torch.square(decoder_frames[:, i, ...] - seqs_y[:, i, ...])) / batch_size
            mae_per_frame[i] = torch.sum(torch.abs(decoder_frames[:, i, ...] - seqs_y[:, i, ...])) / batch_size

        mse_total_frame = torch.sum(torch.mean(torch.square(decoder_frames - seqs_y), dim=(0, 1)))
        mae_total_frame = torch.sum(torch.mean(torch.abs(decoder_frames - seqs_y), dim=(0, 1)))
        val_loss = mse_total_frame + mae_total_frame
        valid_loss_fx = self.loss_fx(decoder_frames, seqs_y)

        if self.hparams.add_video and (self.hparams.add_video and batch_idx % self.hparams.visual_val_steps == 0):
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

        if self.global_step % self.hparams.val_log_steps:
            metrics_pred = add_score_log(self.hparams.predict_class, self.hparams.predict_class_vmax,
                                         seqs_y, decoder_frames, marker="val")
        else:
            metrics_pred = {}

        metrics_pred = {**metrics_pred, "val_loss": val_loss, "val_mse_total": mse_total_frame,
                        "val_mae_total": mae_total_frame, 'valid_loss_fx': valid_loss_fx.item()}
        self.log_dict(metrics_pred, on_step=False, on_epoch=True, prog_bar=True)
        return metrics_pred

    def validation_epoch_end(self, val_outputs_dict):
        epoch_mse_val = 0
        epoch_mae_val = 0
        count = 0
        for out in val_outputs_dict:
            epoch_mse_val += out["val_mse_total"]
            epoch_mae_val += out["val_mae_total"]
            count += 1
        epoch_mae_val = epoch_mae_val / count
        epoch_mse_val = epoch_mse_val / count
        self.log_dict({"epoch_mse_val": epoch_mse_val, "epoch_mae_val": epoch_mae_val})

    def test_step(self, batch, batch_idx):
        seqs_x, seqs_y = batch

        input_flag = torch.ones(1, self.hparams.input_length - 1, 1, 1, 1, device=self.device)
        target_flag = torch.zeros(1, self.hparams.target_length - 1, 1, 1, 1, device=self.device)

        batch_size = seqs_y.shape[0]
        _, decoder_frames, _, _ = self(seqs_x, input_flag, seqs_y, target_flag)
        decoder_frames = torch.clip(decoder_frames, 0, 1)
        if self.hparams.test_save_path:
            for ibatch in range(batch_size):
                dir_test = os.path.join(self.hparams.test_save_path,
                                        "%04d" % (batch_idx * self.hparams.batch_size + ibatch))
                if not os.path.exists(dir_test):
                    os.makedirs(dir_test)
                path_test_prec = os.path.join(dir_test, "prec.pt")
                path_test_radar = os.path.join(dir_test, "radar.pt")
                with open(path_test_prec, "wb") as fp:
                    torch.save(decoder_frames[ibatch, :, 0, :, :], fp)
                with open(path_test_radar, "wb") as fr:
                    torch.save(decoder_frames[ibatch, :, 1, :, :], fr)
        metrics_pred = add_score_log(self.hparams.predict_class, self.hparams.predict_class_vmax,
                                     seqs_y, decoder_frames, marker="test")
        self.log_dict(metrics_pred, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode=self.hparams.lr_scheduler_mode,
                                      patience=self.hparams.lr_scheduler_patience,
                                      factor=self.hparams.lr_scheduler_factor)
        if self.hparams.lr_scheduler:
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler,
                                                             'monitor': self.hparams.lr_scheduler_monitor,
                                                             'frequency': self.hparams.lr_scheduler_frequency}}
        else:
            return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("PredRNN_v2")
        add_parser(parser)
        parser.add_argument('--num_hidden', nargs='+', type=int, default=[128, 128, 128, 128])
        parser.add_argument('--filter_size', type=int, default=5)
        parser.add_argument('--stride', type=int, default=1)
        parser.add_argument('--layer_norm', type=int, default=0)
        parser.add_argument('--lr_scheduler', type=int, default=0)
        parser.add_argument('--lr_scheduler_mode', type=str, default="min")
        parser.add_argument('--lr_scheduler_patience', type=int, default=3)
        parser.add_argument('--lr_scheduler_factor', type=float, default=0.5)
        parser.add_argument('--lr_scheduler_monitor', type=str, default='epoch_mae_val')
        parser.add_argument('--lr_scheduler_frequency', type=int, default=1)
        parser.add_argument('--reverse_scheduled_sampling', type=int, default=1)
        parser.add_argument('--r_sampling_step_1', type=int, default=25000)
        parser.add_argument('--r_sampling_step_2', type=int, default=50000)
        parser.add_argument('--r_exp_alpha', type=float, default=2500)
        parser.add_argument('--scheduled_sampling', type=int, default=0)
        parser.add_argument('--sampling_stop_iter', type=int, default=50000)
        parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)
        parser.add_argument('--param_a', type=float, default=1)
        parser.add_argument('--param_b', type=float, default=2)
        parser.add_argument('--param_c', type=float, default=0.05)
        parser.add_argument('--param_d', type=float, default=0.05)
        parser.add_argument('--share_params', type=int, default=1)
        return parent_parser


class PredRNN_encoder_v2(pl.LightningModule):

    def __init__(self, num_hidden, patched_height, patched_width, filter_size, stride, layer_norm, adapter_num_hidden,
                 num_layers, channels_in):
        super(PredRNN_encoder_v2, self).__init__()
        cell_list_encoder = []
        for i in range(num_layers):
            in_channel = channels_in if i == 0 else num_hidden[i - 1]
            cell_list_encoder.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], patched_height, patched_width, filter_size,
                                       stride, layer_norm)
            )
        self.cell_list_encoder = nn.ModuleList(cell_list_encoder)
        self.conv_last_encoder = nn.Conv2d(num_hidden[num_layers - 1], channels_in,
                                           kernel_size=1, stride=1, padding=0, bias=False)

        self.adapter_encoder = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)
        self.h_t = []
        self.c_t = []
        self.num_layers = num_layers
        self.num_hidden = num_hidden

    def forward(self, input_, first_timestep=False, h_t=None, c_t=None, memory=None):
        """
        ///encoder模块的forward过程
        :param input_:bottom层的输入
        :param first_timestep:是否为第一个
        :return:
        """

        if first_timestep:
            self.h_t = h_t
            self.c_t = c_t
            self.memory = memory

        decouple_loss = 0
        for i, cell in enumerate(self.cell_list_encoder):
            if i == 0:  ## bottom layer
                self.h_t[i], self.c_t[i], self.memory, delta_c, delta_m = cell(input_, self.h_t[i], self.c_t[i],
                                                                               self.memory)
            else:
                self.h_t[i], self.c_t[i], self.memory, delta_c, delta_m = cell(self.h_t[i - 1], self.h_t[i],
                                                                               self.c_t[i],
                                                                               self.memory)

            delta_c_list = F.normalize(self.adapter_encoder(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1),
                                       dim=2)
            delta_m_list = F.normalize(self.adapter_encoder(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1),
                                       dim=2)
            decouple_loss += torch.mean(torch.abs(torch.cosine_similarity(delta_c_list, delta_m_list, dim=2)))

        x_gen_encoder = self.conv_last_encoder(self.h_t[self.num_layers - 1])

        return self.h_t, self.c_t, self.memory, x_gen_encoder, decouple_loss / self.num_layers


class SpatioTemporalLSTMCell(pl.LightningModule):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding,
                          bias=False),
                nn.LayerNorm([num_hidden * 7, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding,
                          bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding,
                          bias=False),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                          bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding,
                          bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding,
                          bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding,
                          bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                          bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = PredRNN_v2.add_model_specific_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)
    m = PredRNN_v2(**dict_args)
    input_ = torch.randn(3, 10, 2, 128, 64)
    mask_input = torch.ones(3, 9, 1, 1, 1)
    output = torch.randn(3, 10, 2, 128, 64)
    mask_output = torch.zeros(3, 9, 1, 1, 1)
    y = m(input_, mask_input, output, mask_output)
