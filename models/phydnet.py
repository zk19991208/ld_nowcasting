import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loss import get_loss_fx
from metrics import add_score_log, add_time_score_log
from utils import add_tensorboard, add_parser
from utils.constrain_moments import K2M
from utils.schedule_sampling import teacher_forcing


class PhyDNet(pl.LightningModule):

    def __init__(self, height, width, input_length, target_length, downscale_factor, learning_rate, loss_fx,
                 input_class, predict_class, predict_class_vmax, add_video, weights_prec, thresholds_prec,
                 weights_radar, thresholds_radar, visual_prec_vmin, visual_prec_vmax, visual_train_steps,
                 visual_val_steps, train_log_steps, val_log_steps, test_save_path, cnn_hidden_size, rnn_input_dim,
                 phycell_hidden_dims, kernel_size_phycell, convlstm_hidden_dims, kernel_size_convlstm,
                 lr_scheduler_mode,
                 lr_scheduler_patience, lr_scheduler_factor, lr_scheduler_monitor, lr_scheduler_frequency,
                 sampling_changing_rate_epoch, param_a, param_b, param_c, param_d, share_params, **kwargs):

        super(PhyDNet, self).__init__()
        self.save_hyperparameters()

        assert height % downscale_factor == 0, "downscale_width should be int!"
        assert width % downscale_factor == 0, "downscale_width should be int!"
        self.num_channels_in = len(input_class)
        self.num_channels_out = len(predict_class)
        self.rnn_cell_height = height // downscale_factor
        self.rnn_cell_width = width // downscale_factor
        if share_params:
            assert self.num_channels_in == self.num_channels_out, "channels in should equal to channels out!"
            self.encoder = EncoderRNN(self.num_channels_in, cnn_hidden_size, self.num_channels_out,
                                      self.rnn_cell_height,
                                      self.rnn_cell_width, rnn_input_dim, phycell_hidden_dims, kernel_size_phycell,
                                      convlstm_hidden_dims, kernel_size_convlstm, downscale_factor)
            self.decoder = self.encoder
        else:
            self.encoder = EncoderRNN(self.num_channels_in, cnn_hidden_size, self.num_channels_out,
                                      self.rnn_cell_height,
                                      self.rnn_cell_width, rnn_input_dim, phycell_hidden_dims, kernel_size_phycell,
                                      convlstm_hidden_dims, kernel_size_convlstm, downscale_factor)
            self.decoder = EncoderRNN(self.num_channels_in, cnn_hidden_size, self.num_channels_out,
                                      self.rnn_cell_height,
                                      self.rnn_cell_width, rnn_input_dim, phycell_hidden_dims, kernel_size_phycell,
                                      convlstm_hidden_dims, kernel_size_convlstm, downscale_factor)

        self.loss_fx = get_loss_fx(loss_fx, predict_class, predict_class_vmax, weights_prec, thresholds_prec,
                                   weights_radar, thresholds_radar)
        self.layers_phys = len(phycell_hidden_dims)
        self.layers_convlstm = len(convlstm_hidden_dims)

    def forward(self, input_tensor, target_tensor, use_teacher_forcing=False, **kwargs):
        batch = input_tensor.shape[0]
        encoder_frames = []
        decoder_frames = []
        h_t = []
        c_t = []
        phys_h_t = []

        for i in range(self.layers_convlstm):
            zeros = torch.zeros(
                [batch, self.hparams.convlstm_hidden_dims[i], self.rnn_cell_height, self.rnn_cell_width],
                device=self.device)
            h_t.append(zeros)
            c_t.append(zeros)

        for i in range(self.layers_phys):
            phys_h_t.append(torch.zeros([batch, self.hparams.rnn_input_dim, self.rnn_cell_height, self.rnn_cell_width],
                                        device=self.device))

        for ei in range(self.hparams.input_length - 1):
            h_t, c_t, phys_h_t, encoder_phys, encoder_conv, output_image = self.encoder(input_tensor[:, ei], ei == 0,
                                                                                        h_t, c_t, phys_h_t)
            encoder_frames.append(output_image)

        h_t, c_t, phys_h_t, encoder_phys, encoder_conv, output_image = self.encoder(input_tensor[:, -1])
        decoder_frames.append(output_image[:, :self.num_channels_out])

        for di in range(self.hparams.target_length - 1):
            if use_teacher_forcing:
                decoder_input = target_tensor[:, di]
            else:
                decoder_input = output_image
            h_t, c_t, phys_h_t, encoder_phys, encoder_conv, output_image = self.decoder(decoder_input, di == 0, h_t,
                                                                                        c_t,
                                                                                        phys_h_t)
            decoder_frames.append(output_image)

        encoder_frames = torch.stack(encoder_frames, dim=1)
        decoder_frames = torch.stack(decoder_frames, dim=1)
        phys_filter_encoder = self.encoder.phycell.cell_list[0].F.conv1.weight
        phys_filter_decoder = self.decoder.phycell.cell_list[0].F.conv1.weight

        return encoder_frames, decoder_frames, phys_filter_encoder, phys_filter_decoder

    def on_train_start(self):
        self.constraints = torch.zeros([self.hparams.phycell_hidden_dims[0], self.hparams.kernel_size_phycell,
                                        self.hparams.kernel_size_phycell], device=self.device)
        ind = 0
        for i in range(self.hparams.kernel_size_phycell):
            for j in range(self.hparams.kernel_size_phycell):
                self.constraints[ind, i, j] = 1
                ind += 1
        self.k2m = K2M([self.hparams.kernel_size_phycell, self.hparams.kernel_size_phycell], self.device)

    def training_step(self, batch, batch_idx):
        seqs_x, seqs_y = batch
        teacher_forcing_bool, eta = teacher_forcing(self.current_epoch, self.hparams.sampling_changing_rate_epoch)
        encoder_frames, decoder_frames, phys_filter_encoder, phys_filter_decoder = self(seqs_x, seqs_y,
                                                                                        teacher_forcing_bool)
        input_loss = self.loss_fx(encoder_frames, seqs_x[:, 1:])
        target_loss = self.loss_fx(decoder_frames, seqs_y)

        constraints_loss_encoder = 0
        constraints_loss_decoder = 0
        for b in range(self.hparams.rnn_input_dim):
            m = self.k2m(phys_filter_encoder[:, b])
            constraints_loss_encoder += F.mse_loss(m, self.constraints)

        for b in range(self.hparams.rnn_input_dim):
            m = self.k2m(phys_filter_decoder[:, b])
            constraints_loss_decoder += F.mse_loss(m, self.constraints)

        loss = self.hparams.param_a * input_loss + self.hparams.param_b * target_loss \
               + self.hparams.param_c * constraints_loss_encoder / self.hparams.rnn_input_dim \
               + self.hparams.param_d * constraints_loss_encoder / self.hparams.rnn_input_dim

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
             "train_constraints_loss_encoder": constraints_loss_encoder, "train_eta": eta,
             "train_constraints_loss_decoder": constraints_loss_decoder}, on_step=True,
            on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seqs_x, seqs_y = batch
        batch_size = seqs_x.shape[0]
        mse_per_frame = torch.zeros(self.hparams.target_length, device=self.device)
        mae_per_frame = torch.zeros(self.hparams.target_length, device=self.device)
        _, decoder_frames, _, _ = self(seqs_x, None, False)
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

        self.log_dict({**metrics_pred}, on_step=False, on_epoch=True, prog_bar=True)
        # self.log_dict(metrics_pred, on_step=False, on_epoch=True, prog_bar=True)
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
        batch_size = seqs_x.shape[0]
        _, decoder_frames, _, _ = self(seqs_x, None, False)
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
        #         #
        #         # path_test_true = os.path.join(dir_test, "truth.npy")
        #         # with open(path_test_true, "wb") as ft:
        #         #     np.save(ft, seqs_y.detach().cpu().numpy()[ibatch, ...])

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
        parser = parent_parser.add_argument_group("PhyDNet")
        add_parser(parser)
        parser.add_argument('--cnn_hidden_size', type=int, default=128)
        parser.add_argument('--rnn_input_dim', type=int, default=128)
        parser.add_argument('--phycell_hidden_dims', nargs="+", type=int, default=[49, ])
        parser.add_argument('--kernel_size_phycell', type=int, default=7)
        parser.add_argument('--convlstm_hidden_dims', nargs="+", type=int, default=[128, 128, 128])
        parser.add_argument('--kernel_size_convlstm', type=int, default=3)
        parser.add_argument('--lr_scheduler_mode', type=str, default="min")
        parser.add_argument('--lr_scheduler_patience', type=int, default=3)
        parser.add_argument('--lr_scheduler_factor', type=float, default=0.5)
        parser.add_argument('--lr_scheduler_monitor', type=str, default='val_loss')
        parser.add_argument('--lr_scheduler_frequency', type=int, default=1)
        parser.add_argument('--sampling_changing_rate_epoch', type=float, default=0.05)
        parser.add_argument('--param_a', type=float, default=1)
        parser.add_argument('--param_b', type=float, default=1)
        parser.add_argument('--param_c', type=float, default=1)
        parser.add_argument('--param_d', type=float, default=1)
        parser.add_argument('--share_params', type=int, default=1)
        return parent_parser


class EncoderRNN(pl.LightningModule):
    def __init__(self, num_channels_in, cnn_hidden_size, num_channels_out, rnn_cell_height, rnn_cell_width,
                 rnn_input_dim,
                 phycell_hidden_dims, kernel_size_phycell, convlstm_hidden_dims, kernel_size_convlstm, downscale=4):

        super(EncoderRNN, self).__init__()
        if downscale == 4:
            self.encoder_E = encoder_4E(nchannels_in=num_channels_in,
                                        nchannels_out=cnn_hidden_size)  # general encoder 64x64x1 -> 32x32x32
            self.decoder_D = decoder_4D(nchannels_in=cnn_hidden_size, nchannels_out=num_channels_out)
        elif downscale == 16:
            self.encoder_E = encoder_16E(nchannels_in=num_channels_in, nchannels_out=cnn_hidden_size)
            self.decoder_D = decoder_16D(nchannels_in=cnn_hidden_size, nchannels_out=num_channels_out)
        elif downscale == 30:
            self.encoder_E = encoder_30E(nchannels_in=num_channels_in, nchannels_out=cnn_hidden_size)
            self.decoder_D = decoder_30D(nchannels_in=cnn_hidden_size, nchannels_out=num_channels_out)
        else:
            raise ("the downscale must in [4, 16, 30]!")
        self.encoder_Ep = encoder_specific(nchannels_in=cnn_hidden_size,
                                           nchannels_out=cnn_hidden_size)  # specific image encoder 32x32x32 -> 16x16x64
        self.encoder_Er = encoder_specific(nchannels_in=cnn_hidden_size, nchannels_out=cnn_hidden_size)
        self.decoder_Dp = decoder_specific(nchannels_in=cnn_hidden_size,
                                           nchannels_out=cnn_hidden_size)  # specific image decoder 16x16x64 -> 32x32x32
        self.decoder_Dr = decoder_specific(nchannels_in=cnn_hidden_size, nchannels_out=cnn_hidden_size)

        self.phycell = PhyCell(input_shape=(rnn_cell_height, rnn_cell_width), input_dim=rnn_input_dim,
                               F_hidden_dims=phycell_hidden_dims,
                               kernel_size=(kernel_size_phycell, kernel_size_phycell))
        self.convcell = ConvLSTM(input_shape=(rnn_cell_height, rnn_cell_width), input_dim=rnn_input_dim,
                                 hidden_dims=convlstm_hidden_dims,
                                 kernel_size=(kernel_size_convlstm, kernel_size_convlstm))

    def forward(self, input_, first_timestep=False, h_t=None, c_t=None, phys_h_t=None):
        input_ = self.encoder_E(input_)

        input_phys = self.encoder_Ep(input_)
        input_conv = self.encoder_Er(input_)

        phys_h_t, output1 = self.phycell(input_phys, first_timestep, phys_h_t)
        (h_t, c_t), output2 = self.convcell(input_conv, first_timestep, h_t, c_t)

        decoded_Dp = self.decoder_Dp(output1[-1])
        decoded_Dr = self.decoder_Dr(output2[-1])

        out_phys = torch.sigmoid(self.decoder_D(decoded_Dp))  # partial reconstructions for vizualization
        out_conv = torch.sigmoid(self.decoder_D(decoded_Dr))

        concat = decoded_Dp + decoded_Dr
        output_image = torch.sigmoid(self.decoder_D(concat))
        return h_t, c_t, phys_h_t, out_phys, out_conv, output_image


class PhyCell_Cell(pl.LightningModule):
    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell_Cell, self).__init__()
        self.input_dim = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.F = nn.Sequential()
        self.F.add_module('conv1',
                          nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, kernel_size=self.kernel_size,
                                    stride=(1, 1), padding=self.padding))
        self.F.add_module('bn1', nn.GroupNorm(7, F_hidden_dim))
        self.F.add_module('conv2',
                          nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim, kernel_size=(1, 1), stride=(1, 1),
                                    padding=(0, 0)))

        self.convgate = nn.Conv2d(in_channels=self.input_dim + self.input_dim,
                                  out_channels=self.input_dim,
                                  kernel_size=(3, 3),
                                  padding=(1, 1), bias=self.bias)

    def forward(self, x, hidden):  # x [batch_size, hidden_dim, height, width]
        combined = torch.cat([x, hidden], dim=1)  # concatenate along channel axis
        combined_conv = self.convgate(combined)
        K = torch.sigmoid(combined_conv)
        hidden_tilde = hidden + self.F(hidden)  # prediction
        next_hidden = hidden_tilde + K * (x - hidden_tilde)  # correction , Haddamard product
        return next_hidden


class PhyCell(pl.LightningModule):
    def __init__(self, input_shape, input_dim, F_hidden_dims, kernel_size):
        super(PhyCell, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = len(F_hidden_dims)
        self.kernel_size = kernel_size
        self.H = []

        cell_list = []
        for i in range(0, self.n_layers):
            cell_list.append(PhyCell_Cell(input_dim=input_dim,
                                          F_hidden_dim=self.F_hidden_dims[i],
                                          kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False, h_t=None):  # input_ [batch_size, 1, channels, width, height]
        # batch_size = input_.data.size()[0]
        if first_timestep:
            #   self.initHidden(batch_size)  # init Hidden at each forward start
            self.H = h_t

        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                self.H[j] = cell(input_, self.H[j])
            else:
                self.H[j] = cell(self.H[j - 1], self.H[j])

        return self.H, self.H


class ConvLSTM_Cell(pl.LightningModule):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTM_Cell, self).__init__()

        self.height, self.width = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)

    # we implement LSTM that process only one timestep
    def forward(self, x, hidden):  # x [batch, hidden_dim, width, height]
        h_cur, c_cur = hidden

        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(pl.LightningModule):
    def __init__(self, input_shape, input_dim, hidden_dims, kernel_size):
        super(ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = len(hidden_dims)
        self.kernel_size = kernel_size
        self.H, self.C = [], []

        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            cell_list.append(ConvLSTM_Cell(input_shape=self.input_shape,
                                           input_dim=cur_input_dim,
                                           hidden_dim=self.hidden_dims[i],
                                           kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False, h_t=None,
                c_t=None):  # input_ [batch_size, 1, channels, width, height]
        if first_timestep:
            self.H = h_t
            self.C = c_t

        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                self.H[j], self.C[j] = cell(input_, (self.H[j], self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j - 1], (self.H[j], self.C[j]))

        return (self.H, self.C), self.H  # (hidden, output)


class dcgan_conv(pl.LightningModule):
    def __init__(self, channels_in, channels_out, stride, kernel_size=3, padding=1):
        super(dcgan_conv, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.GroupNorm(16, channels_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input_):
        return self.down_conv(input_)


class dcgan_upconv(pl.LightningModule):
    def __init__(self, channels_in, channels_out, stride, kernel_size=3, padding=1, output_padding=0):
        super(dcgan_upconv, self).__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channels_in, out_channels=channels_out, kernel_size=kernel_size,
                               stride=stride,
                               padding=padding, output_padding=output_padding),
            nn.GroupNorm(16, channels_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input_):
        return self.up_conv(input_)


class encoder_4E(pl.LightningModule):
    def __init__(self, nchannels_in=1, nchannels_out=64):
        super(encoder_4E, self).__init__()
        # input is (1) x 64 x 64
        self.c1 = dcgan_conv(nchannels_in, nchannels_out // 2, stride=2)  # (32) x 32 x 32
        self.c2 = dcgan_conv(nchannels_out // 2, nchannels_out // 2, stride=1)  # (32) x 32 x 32
        self.c3 = dcgan_conv(nchannels_out // 2, nchannels_out, stride=2)  # (64) x 16 x 16

    def forward(self, input_):
        h1 = self.c1(input_)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h3

class encoder_16E(pl.LightningModule):
    def __init__(self, nchannels_in=1, nchannels_out=128):
        super(encoder_16E, self).__init__()
        # input is (1) x 64 x 64
        self.c1 = dcgan_conv(nchannels_in, nchannels_out // 2, stride=2)  # (32) x 32 x 32
        self.c2 = dcgan_conv(nchannels_out // 2, nchannels_out // 2, stride=2)  # (32) x 32 x 32
        self.c3 = dcgan_conv(nchannels_out // 2, nchannels_out // 2, stride=2)  # (32) x 32 x 32
        self.c4 = dcgan_conv(nchannels_out // 2, nchannels_out, stride=2)  # (64) x 16 x 16

    def forward(self, input_):
        h1 = self.c1(input_)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        return h4

class decoder_16D(pl.LightningModule):
    def __init__(self, nchannels_in=128, nchannels_out=1):
        super(decoder_16D, self).__init__()
        self.upc1 = dcgan_upconv(nchannels_in, nchannels_in // 2, stride=2, output_padding=1)  # (32) x 32 x 32
        self.upc2 = dcgan_upconv(nchannels_in // 2, nchannels_in // 2, stride=2, output_padding=1)  # (32) x 32 x 32
        self.upc3 = dcgan_upconv(nchannels_in // 2, nchannels_in // 2, stride=2, output_padding=1)  # (32) x 32 x 32
        self.upc4 = nn.ConvTranspose2d(in_channels=nchannels_in // 2, out_channels=nchannels_out, kernel_size=(3, 3),
                                       stride=2, padding=1, output_padding=1)  # (nc) x 64 x 64

    def forward(self, input_):
        d1 = self.upc1(input_)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        return d4


class encoder_30E(pl.LightningModule):
    def __init__(self, nchannels_in=1, nchannels_out=64):
        super(encoder_30E, self).__init__()
        # input is (1) x 64 x 64
        self.c1 = dcgan_conv(nchannels_in, nchannels_out // 2, kernel_size=7, stride=5, padding=1)  # (32) x 32 x 32
        self.c2 = dcgan_conv(nchannels_out // 2, nchannels_out // 2, kernel_size=5, stride=3,
                             padding=1)  # (32) x 32 x 32
        self.c3 = dcgan_conv(nchannels_out // 2, nchannels_out, kernel_size=3, stride=2, padding=1)  # (64) x 16 x 16

    def forward(self, input_):
        h1 = self.c1(input_)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h3


class decoder_4D(pl.LightningModule):
    def __init__(self, nchannels_in=64, nchannels_out=1):
        super(decoder_4D, self).__init__()
        self.upc1 = dcgan_upconv(nchannels_in, nchannels_in // 2, stride=2, output_padding=1)  # (32) x 32 x 32
        self.upc2 = dcgan_upconv(nchannels_in // 2, nchannels_in // 2, stride=1)  # (32) x 32 x 32
        self.upc3 = nn.ConvTranspose2d(in_channels=nchannels_in // 2, out_channels=nchannels_out, kernel_size=(3, 3),
                                       stride=2, padding=1, output_padding=1)  # (nc) x 64 x 64

    def forward(self, input_):
        d1 = self.upc1(input_)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        return d3


class decoder_30D(pl.LightningModule):
    def __init__(self, nchannels_in=64, nchannels_out=1):
        super(decoder_30D, self).__init__()
        self.upc1 = dcgan_upconv(nchannels_in, nchannels_in // 2, kernel_size=4, stride=2, padding=1)
        self.upc2 = dcgan_upconv(nchannels_in // 2, nchannels_in // 2, kernel_size=5, stride=3, padding=1)
        self.upc3 = nn.ConvTranspose2d(in_channels=nchannels_in // 2, out_channels=nchannels_out, kernel_size=7, \
                                       stride=5, padding=1)

    def forward(self, input_):
        d1 = self.upc1(input_)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        return d3


class encoder_specific(pl.LightningModule):
    def __init__(self, nchannels_in=64, nchannels_out=64):
        super(encoder_specific, self).__init__()
        self.c1 = dcgan_conv(nchannels_in, nchannels_out, stride=1)  # (64) x 16 x 16
        self.c2 = dcgan_conv(nchannels_out, nchannels_out, stride=1)  # (64) x 16 x 16

    def forward(self, input_):
        h1 = self.c1(input_)
        h2 = self.c2(h1)
        return h2


class decoder_specific(pl.LightningModule):
    def __init__(self, nchannels_in=64, nchannels_out=64):
        super(decoder_specific, self).__init__()
        self.upc1 = dcgan_upconv(nchannels_in, nchannels_in, stride=1)  # (64) x 16 x 16
        self.upc2 = dcgan_upconv(nchannels_in, nchannels_out, stride=1)  # (32) x 32 x 32

    def forward(self, input_):
        d1 = self.upc1(input_)
        d2 = self.upc2(d1)
        return d2


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = PhyDNet.add_model_specific_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)
    m = PhyDNet(**dict_args)
    x = torch.randn(7, 10, 1, 768, 768)
    y = m(x, x, 0)
