import sys
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm

from models import PhyDNet, PredRNN_v2, ConvGRU, Unet, Unet3d, MIM, PhyDNet_ATT
from dataset import RadarPrecReader, RadarPrecWriter, RadarWindReader, RadarWindWriter


def main(args):
    if args.model_name == "predrnn_v2":
        model = PredRNN_v2.load_from_checkpoint(args.weight_path)
    elif args.model_name == "mim":
        model = MIM.load_from_checkpoint(args.weight_path)
    elif args.model_name == "phydnet":
        model = PhyDNet.load_from_checkpoint(args.weight_path, strict=False)
    elif args.model_name == "phydnet_att":
        model = PhyDNet_ATT.load_from_checkpoint(args.weight_path, strict=False)
    elif args.model_name == "unet3d":
        model = Unet3d.load_from_checkpoint(args.weight_path, strict=False)
    elif args.model_name == "convgru":
        model = ConvGRU.load_from_checkpoint(args.weight_path)
    elif args.model_name == "unet":
        model = Unet.load_from_checkpoint(args.weight_path)
    else:
        raise NameError("the model is not in our model zoo!")

    if args.dataset_name == "RadarPrecReader":
        reader = RadarPrecReader(args.radar_dir, args.prec_dir)
        writer = RadarPrecWriter(args.save_radar_dir, args.save_prec_dir)
    elif args.dataset_name == "RadarWindReader":
        reader = RadarWindReader(args.radar_dir, args.wind_dir)
        writer = RadarWindWriter(args.save_radar_dir, args.save_wind_dir)
    else:
        raise NameError("the reader is not in our reader zoo!")

    model.eval()
    with torch.no_grad():
        dir_list = [str(x).rjust(3, "0") for x in range(1, 105, 1)]
        for dir in tqdm(dir_list):
            inputs = reader(dir).cuda()
            _ = torch.zeros((20, 2, 480, 560))
            model = model.cuda()
            _, outputs, _, _ = model(inputs, _, 0)
            outputs = outputs.cpu().numpy()
            outputs = np.clip(outputs, 0, 1)
            outputs[:, :, 0] = outputs[:, :, 0] * 255/35
            outputs[:, :, 1] = outputs[:, :, 1] * 255/70
            writer(dir, outputs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="phydnet_att")
    parser.add_argument("--dataset_name", type=str, default="RadarPrecReader")
    parser.add_argument("--weight_path", type=str, default=r"/home/zhuangxr/work/zengkang/save/RadarPrec/phydnet_att_weighted_l1/weights_epoch=006_HSS_001mm_val=0.520.ckpt")
    parser.add_argument("--radar_dir", type=str, default=r"/home/zhuangxr/work/zengkang/data/TestB1/Radar")
    parser.add_argument("--prec_dir", type=str, default=r"/home/zhuangxr/work/zengkang/data/TestB1/Precip")
    parser.add_argument("--wind_dir", type=str, default=r"/home/zhuangxr/work/zengkang/data/TestB1/Wind")
    parser.add_argument("--save_radar_dir", type=str, default=r"/home/zhuangxr/work/zengkang/save/inference_PrecRadar/Radar")
    parser.add_argument("--save_prec_dir", type=str, default=r"/home/zhuangxr/work/zengkang/save/inference_PrecRadar/Prec")
    parser.add_argument("--save_wind_dir", type=str, default=r"/home/zhuangxr/work/zengkang/save/inference_PrecRadar/Wind")

    args = parser.parse_args()
    model = main(args)
