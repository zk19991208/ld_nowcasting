from argparse import ArgumentParser
from utils import add_tensorboard, add_parser
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from dataset import RadarPrecDataModule, RadarDataModule, MovingMnistDataModule, \
    MovingMnistDataPhysModule
from models import PhyDNet_WRAP, PredRNN_Gan

import torch
torch.set_float32_matmul_precision("medium")


def main(args):
    dict_args = vars(args)
    ###############################################      model  name     ###############################################
    if args.model_name == "predrnn_v2":
        model = PredRNN_v2.load_from_checkpoint(args.weight_path, batch_size=args.batch_size,
                                                test_save_path=args.test_save_path)
    elif args.model_name == "phydnet":
        model = PhyDNet.load_from_checkpoint(args.weight_path, strict=False, batch_size=args.batch_size,
                                             test_save_path=args.test_save_path)
    elif args.model_name == "phydnet_att":
        model = PhyDNet_ATT.load_from_checkpoint(args.weight_path, strict=False, batch_size=args.batch_size,
                                             test_save_path=args.test_save_path)
    elif args.model_name == "phydnet_att_gan":
        model = PhyDNet_ATT_Gan.load_from_checkpoint(args.weight_path, strict=False, batch_size=args.batch_size,
                                             test_save_path=args.test_save_path)
    elif args.model_name == "convgru":
        model = ConvGRU.load_from_checkpoint(args.weight_path, batch_size=args.batch_size, test_save_path=args.test_save_path)
    elif args.model_name == "mim":
        model = MIM.load_from_checkpoint(args.weight_path, batch_size=args.batch_size, test_save_path=args.test_save_path)
    elif args.model_name == "unet":
        model = Unet.load_from_checkpoint(args.weight_path, batch_size=args.batch_size, test_save_path=args.test_save_path)
    elif args.model_name == "unet_gan":
        model = UnetGan.load_from_checkpoint(args.weight_path, batch_size=args.batch_size, test_save_path=args.test_save_path, strict=False)
    elif args.model_name == "unet3d":
        model = Unet3d.load_from_checkpoint(args.weight_path, batch_size=args.batch_size, test_save_path=args.test_save_path)
    elif args.model_name == "unet_fusion":
        model = UnetFusion.load_from_checkpoint(args.weight_path, batch_size=args.batch_size, test_save_path=args.test_save_path)
    elif args.model_name == "unet_fusion_nwp":
        model = UnetFusionNWP.load_from_checkpoint(args.weight_path, batch_size=args.batch_size, test_save_path=args.test_save_path)
    elif args.model_name == "se_resunet":
        model = SeResUNet.load_from_checkpoint(args.weight_path, batch_size=args.batch_size, test_save_path=args.test_save_path)
    elif args.model_name == "se_resunet_gan":
        model = SeResUnetGan.load_from_checkpoint(args.weight_path, batch_size=args.batch_size, test_save_path=args.test_save_path, strict=False)
    elif args.model_name == "sprog":
        model = Sprog()
    elif args.model_name == "phydnet_wrap":
        model = PhyDNet_WRAP.load_from_checkpoint(args.weight_path, strict=False,
                                                                    batch_size=args.batch_size,
                                                                    test_save_path=args.test_save_path)
    elif args.model_name == "predrnn_gan":
        model = PredRNN_Gan.load_from_checkpoint(args.weight_path, batch_size=args.batch_size, test_save_path=args.test_save_path)

    else:
        raise NameError("the model is not in our model zoo!")
    ###############################################      dataset  name     #############################################
    ######################   Nowcasting mulC
    if args.dataset_name == "RadarPrecDataModule":
        data = RadarPrecDataModule(**dict_args)
    ######################   Nowcasting 1C
    elif args.dataset_name == "RadarDataModule":
        data = RadarDataModule(**dict_args)
    ######################   MovingMnist
    elif args.dataset_name == "MovingMnistDataModule":
        data = MovingMnistDataModule(**dict_args)
    elif args.dataset_name == "MovingMnistDataPhysModule":
        data = MovingMnistDataPhysModule(**dict_args)
    else:
        raise NameError("the dataset is not in our dataset zoo!")

    ###############################################      train  args     #############################################

    model.freeze()
    model.eval()
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.tensorboard_save_path,
                                             name=args.tensorboard_exp_name)
    trainer = Trainer(gpus=args.gpus, logger=tb_logger, strategy='ddp')
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="predrnn_v2_share")
    parser.add_argument("--dataset_name", type=str, default="RadarDataModule")
    parser.add_argument("--tensorboard_save_path", type=str, default=None)
    parser.add_argument("--tensorboard_exp_name", type=str, default=None)
    parser.add_argument("--weight_path", type=str, default=None)
    parser.add_argument("--train_file", type=str, default="/dev/shm/EastChina/dataset_train.csv")
    parser.add_argument("--val_file", type=str, default="/dev/shm/EastChina/dataset_val.csv")
    parser.add_argument("--test_file", type=str, default="/home/zhengyu/data/EastChina/dataset_test_less.csv")
    parser.add_argument("--radar_dir", type=str, default="/dev/shm/EastChina/Radar")
    parser.add_argument("--prec_dir", type=str, default="/dev/shm/EastChina/Precip")
    parser.add_argument("--root_dir", type=str, default="/dev/shm/EastChina")
    parser.add_argument("--test_save_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_works", type=int, default=8)
    parser.add_argument("--pin_memory", type=int, default=1)
    parser.add_argument("--gpus", type=str, default="0")
    args = parser.parse_args()
    model = main(args)
