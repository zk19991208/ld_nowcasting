"""Lightning 训练入口：图像空间模型、雷达 AE 与潜空间 Predictor。

用法（在 ld_pred 项目根目录下，按需激活 PyTorch 环境）：
  python train.py --model_name earthformer --dataset_name MovingMnistDataPhysModule ...

earthformer 与 SimVP 共用同一 DataModule 约定：batch 为 (seqs_x, seqs_y)，
张量形状均为 (B, T, C, H, W)；EarthformerLit 内部会 permute 为 (B, T, H, W, C) 喂 CuboidTransformerModel。
height/width、input_length/target_length 须与模型 yaml 中 input_shape/target_shape 一致（见 --earthformer_oc_file）。
"""
from argparse import ArgumentParser
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from dataset import SingleRadarDataModule, MovingMnistDataModule, MovingMnistDataPhysModule

from models import RadarAutoEncoderLit, RadarLatentPredictorLit, SimVP_Lit
from models.convlstm import ConvLSTM
from models.earthformer_lit import EarthformerLit

import torch
torch.set_float32_matmul_precision("medium")


def _arg_int_bool(s) -> int:
    """兼容 0/1 与 YAML/run_train_yaml 传出的 true、false 字符串。"""
    if isinstance(s, bool):
        return int(s)
    t = str(s).lower().strip()
    if t in ("true", "yes", "1"):
        return 1
    if t in ("false", "no", "0", ""):
        return 0
    return int(t)


def _trainer_accelerator_devices_strategy(accelerator: str, devices, gpus: str):
    """供 inference 脚本解析设备：返回 (acc_lower, dev_spec, None)。

    dev_spec：GPU 时为与 ``--gpus`` 或 ``--devices`` 等价的字符串；CPU 为 ``None``；NPU 为 ``"0"``。
    """
    acc = (accelerator or "gpu").lower()
    if acc == "cpu":
        return acc, None, None
    if acc == "npu":
        return acc, "0", None
    dev = devices if devices is not None else gpus
    return "gpu", dev, None


def main(args):
    dict_args = vars(args)
    ###############################################      model  name     ###############################################
    # PyTorch Lightning 2.x：须在「类」上调用 load_from_checkpoint，禁止 Model(**args).load_from_checkpoint(...)
    if args.model_name == "simvp":
        if args.weight_path:
            model = SimVP_Lit.load_from_checkpoint(
                args.weight_path,
                strict=False,
                map_location="cpu",
                batch_size=args.batch_size,
                test_save_path=args.test_save_path,
                freeze_encoder=args.freeze_encoder,
                freeze_mid=args.freeze_mid,
                freeze_decoder=args.freeze_decoder,
            )
        else:
            model = SimVP_Lit(**dict_args)
    elif args.model_name == "convlstm":
        if args.weight_path:
            model = ConvLSTM.load_from_checkpoint(
                args.weight_path,
                strict=False,
                map_location="cpu",
                batch_size=args.batch_size,
                test_save_path=args.test_save_path,
            )
        else:
            model = ConvLSTM(**dict_args)
    elif args.model_name == "earthformer":
        if args.weight_path:
            model = EarthformerLit.load_from_checkpoint(
                args.weight_path,
                strict=False,
                map_location="cpu",
                batch_size=args.batch_size,
                test_save_path=args.test_save_path,
            )
        else:
            model = EarthformerLit(**dict_args)
    elif args.model_name == "radar_autoencoder":
        if args.weight_path:
            model = RadarAutoEncoderLit.load_from_checkpoint(
                args.weight_path,
                strict=False,
                map_location="cpu",
                batch_size=args.batch_size,
                test_save_path=args.test_save_path,
            )
        else:
            model = RadarAutoEncoderLit(**dict_args)
    elif args.model_name == "radar_latent_predictor":
        if args.weight_path:
            model = RadarLatentPredictorLit.load_from_checkpoint(
                args.weight_path,
                strict=False,
                map_location="cpu",
            )
        else:
            model = RadarLatentPredictorLit(**dict_args)
    else:
        raise NameError(
            "本脚本支持 simvp / convlstm / earthformer / radar_autoencoder / "
            "radar_latent_predictor，"
            f"当前 model_name={args.model_name!r}。"
        )


    ###############################################      dataset  name     #############################################
    ######################   Nowcasting mulC
    if  args.dataset_name == "SingleRadarDataModule":
        data = SingleRadarDataModule(**dict_args)
    ######################   MovingMnist
    elif args.dataset_name == "MovingMnistDataModule":
        data = MovingMnistDataModule(**dict_args)
    elif args.dataset_name == "MovingMnistDataPhysModule":
        data = MovingMnistDataPhysModule(**dict_args)
    else:
        raise NameError("the dataset is not in our dataset zoo!")

    ###############################################      train  args     #############################################
    save_callback = ModelCheckpoint(
        monitor=args.save_monitor,
        dirpath=args.save_dirpath,
        filename=args.save_filename,
        save_top_k=args.save_top_k,
        mode=args.save_mode,
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.tensorboard_save_path,
                                             name=args.tensorboard_exp_name)
    tqdm_cb = TQDMProgressBar(refresh_rate=args.refresh_rate)
    callbacks = [save_callback, tqdm_cb]
    if args.early_stopping_on:
        early_stop_callback = EarlyStopping(
            monitor=args.early_stopping_monitor,
            patience=args.early_stopping_patience,
            verbose=args.early_stopping_verbose,
            mode=args.early_stopping_mode
        )
        callbacks.append(early_stop_callback)
    if args.resume_path:
        resume_path = args.resume_path
    else:
        resume_path = None

    if resume_path and int(args.resume_weights_only):
        ckpt_obj = torch.load(resume_path, map_location="cpu")
        sd = ckpt_obj.get("state_dict", ckpt_obj)
        inc = model.load_state_dict(sd, strict=False)
        mk = getattr(inc, "missing_keys", ()) or ()
        uk = getattr(inc, "unexpected_keys", ()) or ()
        if mk:
            print(f"[resume_weights_only] missing_keys（前 5）: {mk[:5]}")
        if uk:
            print(f"[resume_weights_only] unexpected_keys（前 5）: {uk[:5]}")
        print(f"[resume_weights_only] 已从 {resume_path} 加载权重；优化器从零开始。")
        resume_path = None

    if args.with_GAN:
        find_unused_parameters = True
    else:
        find_unused_parameters = False
    

    # trainer = Trainer(logger=tb_logger,
    #                   strategy=DDPPlugin(find_unused_parameters=find_unused_parameters),
    #                   accelerator="gpu",
    #                   devices=args.gpus,
    #                   val_check_interval=args.check_val_rate,
    #                   callbacks=callbacks,
    #                   （PL2：续训在 fit(ckpt_path=...)）
    #                   max_epochs=args.max_epochs,
    #                   )
    
    trainer_devices = args.devices if args.devices is not None else args.gpus
    trainer = Trainer(
        logger=tb_logger,
        strategy="ddp",
        accelerator=args.accelerator,
        devices=trainer_devices,
        val_check_interval=args.check_val_rate,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    # Lightning 2.x：断点续训用 fit(ckpt_path=...)，勿再传 Trainer(resume_from_checkpoint=...)
    trainer.fit(model, datamodule=data, ckpt_path=resume_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="simvp")
    parser.add_argument("--dataset_name", type=str, default="RadarHimaWindDataModule")
    parser.add_argument("--gpus", type=str, default='-1')
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="Trainer accelerator：gpu（含曙光 DCU 上 DTK PyTorch）、cpu、npu（昇腾）。DCU 请勿使用 npu。",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Trainer devices 数量或索引；若省略则沿用 --gpus（字符串，如 1 或 0,1）。",
    )
    parser.add_argument("--tensorboard_save_path", type=str, default=r'/home/data/MSG/pl_GAN/work/log')
    parser.add_argument("--tensorboard_exp_name", type=str, default='lighting')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--pin_memory",  type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        help="Lightning precision，例如 32-true 或 16-mixed。",
    )
    parser.add_argument("--gradient_clip_val", type=float, default=0.0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--train_file", type=str, default='Data/tianchi/dataset_train.csv')
    parser.add_argument("--val_file", type=str, default='Data/tianchi/dataset_testA.csv')
    parser.add_argument("--test_file", type=str, default='Data/tianchi/dataset_testA.csv')
    parser.add_argument("--radar_dir", type=str, default='Data/tianchi/Train/Radar')
    parser.add_argument(
        "--packed_sequence_file",
        type=int,
        default=0,
        help="SingleRadar：1 时 CSV 为单列条带 PNG 路径（每样本一条横向长条）；0 时为多列 PNG。",
    )
    parser.add_argument(
        "--packed_event_npy",
        type=int,
        default=0,
        help="SingleRadar：1 时 CSV 为事件索引 + dBZ/data_dir_*/frame_*.npy（见 preprocess/huadong_cr_6min/05_export_event_frames_npy.py）；与 --packed_sequence_file 互斥。",
    )
    parser.add_argument(
        "--radar_dir_val",
        type=str,
        default=None,
        help="验证集雷达根目录；默认与 --radar_dir 相同。事件条带模式下常设为 .../event_strip/val。",
    )
    parser.add_argument(
        "--radar_dir_test",
        type=str,
        default=None,
        help="测试集雷达根目录；默认与 --radar_dir 相同。",
    )
    parser.add_argument(
        "--val_sample_interval",
        type=int,
        default=1,
        help="SingleRadar：验证集每隔多少条样本取 1 条（1=全量；3 表示索引 0,3,6,...）。",
    )
    parser.add_argument("--prec_dir", type=str, default='Data/tianchi/Train/Precip')
    parser.add_argument("--root_dir", type=str, default="tianchi")
    parser.add_argument("--early_stopping_on", type=int, default=0)
    parser.add_argument("--early_stopping_monitor", type=str, default="val_loss")
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--early_stopping_verbose", type=int, default=0)
    parser.add_argument("--early_stopping_mode", type=str, default="min")
    parser.add_argument("--save_monitor", type=str, default="valid_loss_fx")
    parser.add_argument("--save_dirpath", type=str, default='/home/data/MSG/pl_GAN/work/save')
    parser.add_argument("--save_filename", type=str, default='weights-{epoch:03d}-{valid_loss_fx:.3f}', )
    parser.add_argument("--save_top_k", type=int, default=50)
    parser.add_argument("--save_mode", type=str, default="min")
    parser.add_argument("--refresh_rate", type=float, default=30)
    parser.add_argument("--check_val_rate", type=float, default=0.5)
    parser.add_argument("--resume_path", type=str, default="")
    parser.add_argument(
        "--resume_weights_only",
        type=_arg_int_bool,
        default=0,
        help=(
            "1：仅从 resume_path 加载模型权重（state_dict），不恢复优化器/scheduler/epoch。"
            "冻结 Submodule 后与旧 ckpt 优化器形状不一致时必须为 1，否则会报 optimizer load_state_dict 失败。"
        ),
    )
    parser.add_argument("--weight_path", type=str, default=None)
    # add_parser(parser)


    temp_args, _ = parser.parse_known_args()
    if temp_args.model_name == "simvp":
        parser = SimVP_Lit.add_model_specific_args(parser)
    elif temp_args.model_name == "convlstm":
        parser = ConvLSTM.add_model_specific_args(parser)
    elif temp_args.model_name == "earthformer":
        parser = EarthformerLit.add_model_specific_args(parser)
    elif temp_args.model_name == "radar_autoencoder":
        parser = RadarAutoEncoderLit.add_model_specific_args(parser)
    elif temp_args.model_name == "radar_latent_predictor":
        parser = RadarLatentPredictorLit.add_model_specific_args(parser)
    else:
        raise NameError(
            "本脚本支持 simvp / convlstm / earthformer / radar_autoencoder / "
            "radar_latent_predictor，"
            f"当前 model_name={temp_args.model_name!r}。"
        )
    args = parser.parse_args()
    main(args)
