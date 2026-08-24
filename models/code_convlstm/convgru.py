import os

import pytorch_lightning as pl
import torch
import torch.nn as nn

from loss import get_loss_fx
from metrics import add_score_log
from utils import add_tensorboard, add_parser


class ConvGRU(pl.LightningModule):

    def __init__(self, height, width, input_length, target_length, downscale_factor, learning_rate, loss_fx,
                 input_class, predict_class, predict_class_vmax, add_video, weights_prec, thresholds_prec,
                 weights_radar, thresholds_radar, visual_prec_vmin, visual_prec_vmax, visual_train_steps,
                 visual_val_steps, train_log_steps, val_log_steps, test_save_path, convlstm_hidden_dims,
                 lr_scheduler_gamma, **kwargs):

        super(ConvGRU, self).__init__()
        self.save_hyperparameters()
        assert height == width, "height should be equal to width!"
        self.num_channels_in = len(input_class)
        self.num_channels_out = len(predict_class)
        self.encoder = EncoderConv(width, self.num_channels_in)
        self.decoder = DecoderConv(width, self.num_channels_out)
        self.loss_fx = get_loss_fx(loss_fx, predict_class, predict_class_vmax, weights_prec, thresholds_prec,
                                   weights_radar, thresholds_radar)
        self.rnn_cell_width = [width // 5, width // 15, width // 30]

    def forward(self, input_tensor):
        batch = input_tensor.shape[0]
        decoder_frames = []
        h_t = []
        for i in range(3):
            zeros = torch.zeros(
                [batch, self.hparams.convlstm_hidden_dims[i], self.rnn_cell_width[i], self.rnn_cell_width[i]],
                device=self.device)
            h_t.append(zeros)

        for ei in range(self.hparams.input_length):
            output_image, h_t = self.encoder(input_tensor[:, ei], ei == 0, h_t)
        h_t = h_t[::-1]
        for di in range(self.hparams.target_length):
            output_image, h_t = self.decoder(di == 0, h_t)
            decoder_frames.append(output_image)

        decoder_frames = torch.stack(decoder_frames, dim=1)
        return decoder_frames

    def training_step(self, batch, batch_idx):
        seqs_x, seqs_y = batch
        decoder_frames = self(seqs_x)
        loss = self.loss_fx(decoder_frames, seqs_y)
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

        self.log_dict({**metrics_pred, "train_loss": loss, }, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        seqs_x, seqs_y = batch
        batch_size = seqs_y.shape[0]
        mse_per_frame = torch.zeros(self.hparams.target_length, device=self.device)
        mae_per_frame = torch.zeros(self.hparams.target_length, device=self.device)

        decoder_frames = self(seqs_x)
        decoder_frames = torch.clip(decoder_frames, 0, 1)
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

        if self.global_step % self.hparams.val_log_steps:
            metrics_pred = add_score_log(self.hparams.predict_class, self.hparams.predict_class_vmax,
                                         seqs_y, decoder_frames, marker="val")
        else:
            metrics_pred = {}

        metrics_pred = {**metrics_pred, "val_loss": val_loss, "val_mse_total": mse_total_frame,
                        "val_mae_total": mae_total_frame, 'valid_loss_fx': valid_loss_fx.item()}
        self.log_dict(metrics_pred, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):

        seqs_x, seqs_y = batch
        batch_size = seqs_y.shape[0]
        mse_per_frame = torch.zeros(self.hparams.target_length, device=self.device)
        mae_per_frame = torch.zeros(self.hparams.target_length, device=self.device)

        decoder_frames = self(seqs_x)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-8)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.lr_scheduler_gamma)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ConvGRU")
        add_parser(parser)
        parser.add_argument('--convlstm_hidden_dims', nargs="+", type=int, default=[64, 128, 128])
        parser.add_argument('--lr_scheduler_gamma', type=float, default=0.98)
        return parent_parser


class ConvGRU_Cell(nn.Module):
    """搭建convGRU的基本单元
    \begin{array}{ll}
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
    \end{array}
    """

    def __init__(self, input_channel, input_shape, hidden_channel, kernel_size):
        """
        GRU的单元
        :param input_channel: 输入的x的channel数
        :param hidden_channel: 隐藏层的channel数，可以等同GRU的hidden size
        :param kernel_size: 卷积的单元的大小，如3
        :param input_shape: 输入图像的（Height、width）
        """
        super(ConvGRU_Cell, self).__init__()
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.height, self.width = input_shape
        self.conv_hidden = nn.Conv2d(self.hidden_channel, 3 * self.hidden_channel, \
                                     kernel_size=kernel_size, padding=int((kernel_size - 1) / 2))  ##卷积隐藏层
        self.conv_input = nn.Conv2d(self.input_channel, 3 * self.hidden_channel, kernel_size=kernel_size, \
                                    padding=int((kernel_size - 1) / 2))  ##卷积输入层

    def forward(self, input_x, hidden_state=None):
        """
        :param input_x:网络的输入x  (nbatch, ninput_channel, nheight, nwidth)
        :param hidden_state:  (nbatch, nhidden_channel, nheight, nwidth)
        :return:
        """

        if hidden_state is None:
            self.hidden_state = torch.zeros((input_x.size(0), self.hidden_channel, self.height,
                                             self.width), dtype=input_x.dtype, device=input_x.device)
        else:
            self.hidden_state = hidden_state

        combined_hidden = self.conv_hidden(self.hidden_state)
        combined_input = self.conv_input(input_x)

        self.Wir, self.Wiz, self.Win = torch.split(combined_input, self.hidden_channel, dim=1)
        self.Whr, self.Whz, self.Whn = torch.split(combined_hidden, self.hidden_channel, dim=1)

        r = torch.sigmoid(self.Wir + self.Whr)
        z = torch.sigmoid(self.Wiz + self.Whz)
        n = torch.tanh(self.Win + r * self.Whn)
        self.hidden_state = (1 - z) * n + z * self.hidden_state

        return self.hidden_state, self.hidden_state


class EncoderConv(pl.LightningModule):

    def __init__(self, width, channels_in):
        """
        GRU的单元
        :param input_channel: 输入的x的channel数
        :param hidden_channel: 隐藏层的channel数，可以等同GRU的hidden size
        :param kernel_size: 卷积的单元的大小，如3
        :param input_shape: 输入图像的（Height、width）
        :param nlayer: RNN的层数
        """
        super(EncoderConv, self).__init__()
        self.RNN_Layers = nn.ModuleList([
            nn.Conv2d(in_channels=channels_in, out_channels=32, kernel_size=7, stride=5, padding=1),
            nn.GroupNorm(16, 32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ConvGRU_Cell(input_channel=32, input_shape=(width // 5, width // 5), hidden_channel=64, kernel_size=3),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ConvGRU_Cell(input_channel=128, input_shape=(width // 15, width // 15), hidden_channel=128, kernel_size=3),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ConvGRU_Cell(input_channel=128, input_shape=(width // 30, width // 30), hidden_channel=128, kernel_size=3),
        ])

    def forward(self, input_x, first_timestep=False, h_t=None):
        """
        初始的输入和初始的隐藏层
        :param input:  (nseqs_len, nbatch, nchannel, nheight, nwidth)
        :param hidden_state_old: (h_0, ) , h_0: (nlayers, nbatch, nchannel, nheight, nwidth) ,
        :return: output, (out_CRNN, h_n_CRNN)
        out_CRNN [nseqs, nbatch, nchannel, nheight]
        h_n_CRNN list [nrnnlayer]
        """
        if first_timestep:
            self.h_t = h_t
        irnn_layer = 0
        for ilayer in self.RNN_Layers:
            if len(ilayer._modules) == 2:  ##判断是不是ConvRNN层
                input_x, self.h_t[irnn_layer] = ilayer(input_x, self.h_t[irnn_layer])
                irnn_layer += 1
            else:
                input_x = ilayer(input_x)
        return input_x, self.h_t


class DecoderConv(pl.LightningModule):

    def __init__(self, width, channels_out):
        """
        预报网络
        :param input_channel:输入的channel
        :param input_shape: 输入图像的shape
        :param hidden_channel: 隐藏层的channel
        :param kernel_size: 卷积核的大小
        :param output_channel: 输出的channel
        :param nlayer:RNN的层数
        """
        super(DecoderConv, self).__init__()

        self.channels_out = channels_out
        self.width = width
        diff1 = width // 15 - width // 30 * 2
        diff2 = width // 5 - width // 15 * 3
        diff3 = width - width // 5 * 5

        self.RNN_Layers = nn.ModuleList([
            ConvGRU_Cell(input_channel=128, input_shape=(width // 30, width // 30), hidden_channel=128, kernel_size=3),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReflectionPad2d((diff1 // 2, diff1 - diff1 // 2, diff1 // 2, diff1 - diff1 // 2)),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ConvGRU_Cell(input_channel=128, input_shape=(width // 15, width // 15), hidden_channel=128, kernel_size=3),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=3, padding=1),
            nn.ReflectionPad2d((diff2 // 2, diff2 - diff2 // 2, diff2 // 2, diff2 - diff2 // 2)),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ConvGRU_Cell(input_channel=64, input_shape=(width // 5, width // 5), hidden_channel=64, kernel_size=3),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=7, stride=5, padding=1),
            nn.ReflectionPad2d((diff3 // 2, diff3 - diff3 // 2, diff3 // 2, diff3 - diff3 // 2)),
            nn.GroupNorm(16, 32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=channels_out, kernel_size=1, stride=1, padding=0)])

    def forward(self, first_timestep=False, h_t=None):
        """
        :param hidden_state: list [nrnn_layer], [nbatchs, nchannel, nheight, nwidth], encoder's output 需要翻转一下
        :return: [target_seq_len, nbatchs, nchannel, nheight, nwidth]
        """

        if first_timestep:
            self.h_t = h_t
            self.batch_size, _, _, _ = self.h_t[0].shape

        input_x = torch.zeros(self.batch_size, 128, self.width // 30, self.width // 30, device=self.device)
        irnn_layer = 0
        for ilayer in self.RNN_Layers:
            if len(ilayer._modules) == 2:  ##判断是不是ConvRNN层
                input_x, self.h_t[irnn_layer] = ilayer(input_x, self.h_t[irnn_layer])
                irnn_layer += 1
            else:
                input_x = ilayer(input_x)
        return input_x, self.h_t


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    ConvGRU.add_model_specific_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)
    model = ConvGRU(**dict_args)
    x = torch.randn(7, 10, 2, 64, 64)
    y = model(x)
