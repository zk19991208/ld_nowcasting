import sys

sys.path.append("../")
from pytorch_lightning import Trainer
from models import PhyDNet_share, PhyDNet, PredRNN_v2, PredRNN_v2_share, ConvGRU, Unet
from argparse import ArgumentParser
from dataset import RadarPrecDataModule, RadarDataModule, PrecDataModule, MovingMnistDataModule, \
    MovingMnistDataPhysModule


def main(args):
    dict_args = vars(args)
    if args.model_name == "predrnn_v2":
        model = PredRNN_v2(**dict_args)
    elif args.model_name == "predrnn_v2_share":
        model = PredRNN_v2_share(**dict_args)
    elif args.model_name == "phydnet":
        model = PhyDNet(**dict_args)
    elif args.model_name == "phydnet_share":
        model = PhyDNet_share(**dict_args)
    elif args.model_name == "convgru":
        model = ConvGRU(**dict_args)
    elif args.model_name == "unet":
        model = Unet(**dict_args)
    else:
        raise NameError("the model is not in our model zoo!")

    if args.dataset_name == "RadarPrecDataModule":
        data = RadarPrecDataModule(**dict_args)
    elif args.dataset_name == "RadarDataModule":
        data = RadarDataModule(**dict_args)
    elif args.dataset_name == "PrecDataModule":
        data = PrecDataModule(**dict_args)
    elif args.dataset_name == "MovingMnistDataModule":
        data = MovingMnistDataModule(**dict_args)
    elif args.dataset_name == "MovingMnistDataPhysModule":
        data = MovingMnistDataPhysModule(**dict_args)
    else:
        raise NameError("the dataset is not in our dataset zoo!")

    trainer = Trainer(auto_lr_find=True, gpus=args.gpus)
    # trainer = Trainer(auto_scale_batch_size=True, gpus=args.gpus)
    trainer.tune(model, datamodule=data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="phydnet_share")
    parser.add_argument("--dataset_name", type=str, default="RadarPrecDataModule")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_works", type=int, default=8)
    parser.add_argument("--pin_memory", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--train_file", type=str, default="/dev/shm/EastChina/dataset_train.csv")
    parser.add_argument("--val_file", type=str, default="/dev/shm/EastChina/dataset_val.csv")
    parser.add_argument("--test_file", type=str, default="/dev/shm/EastChina/dataset_test.csv")
    parser.add_argument("--radar_dir", type=str, default="/dev/shm/EastChina/Radar")
    parser.add_argument("--prec_dir", type=str, default="/dev/shm/EastChina/Precip")
    parser.add_argument("--root_dir", type=str, default="/dev/shm/EastChina")
    temp_args, _ = parser.parse_known_args()
    # let the model add what it wants
    if temp_args.model_name == 'predrnn_v2':
        parser = PredRNN_v2.add_model_specific_args(parser)
    elif temp_args.model_name == 'predrnn_v2_share':
        parser = PredRNN_v2_share.add_model_specific_args(parser)
    elif temp_args.model_name == "phydnet":
        parser = PhyDNet.add_model_specific_args(parser)
    elif temp_args.model_name == 'phydnet_share':
        parser = PhyDNet_share.add_model_specific_args(parser)
    elif temp_args.model_name == "convgru":
        parser = ConvGRU.add_model_specific_args(parser)
    elif temp_args.model_name == 'unet':
        parser = Unet.add_model_specific_args(parser)
    else:
        pass
    args = parser.parse_args()
    main(args)
