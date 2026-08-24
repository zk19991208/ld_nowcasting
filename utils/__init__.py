from .tbcolor import get_rgb_img_4d


def Z2R(dBZ, a=200, b=1.6):
    return (10 ** (dBZ / 10.) / a) ** (1 / b) / 10.


def add_tensorboard(logger, input_class, predict_class, predict_class_vmax, seqs_x, decoder_frames, seqs_y,
                    visual_prec_vmin, visual_prec_vmax, steps, marker="train"):
    if 0 in predict_class:
        prec_index = predict_class.index(0)
        prec_index_input = input_class.index(0)
        prec_unorm = predict_class_vmax[prec_index]
        logger.experiment.add_video("%s/input" % marker,
                                    get_rgb_img_4d(seqs_x[:, :, prec_index_input, :, :] * prec_unorm,
                                                   visual_prec_vmin, visual_prec_vmax), steps)
        logger.experiment.add_video("%s/predict" % marker,
                                    get_rgb_img_4d(decoder_frames[:, :, prec_index, :, :] * prec_unorm,
                                                   visual_prec_vmin, visual_prec_vmax), steps)
        logger.experiment.add_video("%s/ground_truth" % marker,
                                    get_rgb_img_4d(seqs_y[:, :, prec_index, :, :] * prec_unorm,
                                                   visual_prec_vmin, visual_prec_vmax), steps)
    if 1 in predict_class:
        radar_index = predict_class.index(1)
        radar_index_input = input_class.index(1)
        logger.experiment.add_video("%s/input_radar" % marker,
                                    seqs_x[:, :, radar_index_input:radar_index_input + 1, :, :],
                                    steps)
        logger.experiment.add_video("%s/predict_radar" % marker,
                                    decoder_frames[:, :, radar_index:radar_index + 1, :, :],
                                    steps)
        logger.experiment.add_video("%s/ground_truth_radar" % marker, seqs_y[:, :, radar_index:radar_index + 1, :, :],
                                    steps)


def add_parser(parser):
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=560)
    parser.add_argument('--input_length', type=int, default=20)
    parser.add_argument('--target_length', type=int, default=30)
    parser.add_argument('--downscale_factor', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--loss_fx', type=str, default="weighted_l1")
    parser.add_argument('--input_class', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--predict_class', nargs='+', type=int, default=[0, 1])
    parser.add_argument("--predict_class_vmax", nargs='+', type=float, default=[10, 70])
    parser.add_argument('--add_video', type=int, default=0)
    parser.add_argument('--weights_prec', nargs='+', type=float, default=[1, 1, 2.5, 5, 10, 20])
    parser.add_argument('--thresholds_prec', nargs='+', type=float, default=[0.1, 0.2, 0.5, 1, 2])
    parser.add_argument('--weights_radar', nargs='+', type=float, default=[0.5, 1, 2.5, 5, 10, 15])
    parser.add_argument('--thresholds_radar', nargs='+', type=float, default=[15, 25, 35, 45, 50])
    parser.add_argument("--visual_prec_vmin", type=float, default=-0.2)
    parser.add_argument("--visual_prec_vmax", type=float, default=5)
    parser.add_argument("--visual_train_steps", type=int, default=100)
    parser.add_argument("--visual_val_steps", type=int, default=5)
    parser.add_argument("--train_log_steps", type=int, default=50)
    parser.add_argument("--val_log_steps", type=int, default=1)
    parser.add_argument('--test_save_path', type=str, default="")
    ### GAN
    parser.add_argument('--with_GAN', action='store_true')
    parser.add_argument('--with_VGG', action='store_true')
    parser.add_argument('--with_frequency', action='store_true')
    parser.add_argument('--gan_str', type=str, default="PiexlGAN")
    parser.add_argument('--gan_loss_weight', type=float, default=1)
    parser.add_argument('--vgg_loss_weight', type=float, default=500000)
    parser.add_argument(
        '--freq_loss_weight',
        type=float,
        default=0.0,
        help='>0 时叠加 MultiScaleFrequencyLoss（SimVP 等在模型内组合）；0 表示关闭。',
    )
    parser.add_argument('--freq_loss_alpha', type=float, default=1.0)
    parser.add_argument('--freq_loss_scales', nargs='+', type=int, default=[1, 2, 4])