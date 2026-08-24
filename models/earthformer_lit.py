# Earthformer CuboidTransformer 的 Lightning 封装：与 transfer/train.py、MovingMNIST DataModule 对齐。
# 数据为 (B,T,C,H,W)，内部转为 Earthformer 所需的 (B,T,H,W,C)。
# 用法：train.py --model_name earthformer --dataset_name MovingMnistDataPhysModule ...（需 omegaconf、einops）
# 可选：--earthformer_oc_file 指向自定义 yaml（需含 model: 段，结构同 earthformer_default_mmnist.yaml）。

from __future__ import annotations

import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from loss import get_loss_fx
from utils import add_parser

from models.earthformer_vendor.earthformer.cuboid_transformer import CuboidTransformerModel

_DEFAULT_CFG = Path(__file__).resolve().parent / "earthformer_default_mmnist.yaml"


def _build_cuboid_from_model_cfg(model_cfg: dict) -> CuboidTransformerModel:
    num_blocks = len(model_cfg["enc_depth"])
    sp = model_cfg["self_pattern"]
    csp = model_cfg["cross_self_pattern"]
    cp = model_cfg["cross_pattern"]
    enc_attn_patterns = [sp] * num_blocks if isinstance(sp, str) else list(sp)
    dec_self_attn_patterns = [csp] * num_blocks if isinstance(csp, str) else list(csp)
    dec_cross_attn_patterns = [cp] * num_blocks if isinstance(cp, str) else list(cp)

    block_units = model_cfg.get("block_units", None)

    return CuboidTransformerModel(
        input_shape=tuple(model_cfg["input_shape"]),
        target_shape=tuple(model_cfg["target_shape"]),
        base_units=model_cfg["base_units"],
        block_units=block_units,
        scale_alpha=model_cfg["scale_alpha"],
        enc_depth=model_cfg["enc_depth"],
        dec_depth=model_cfg["dec_depth"],
        enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
        dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
        dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
        downsample=model_cfg["downsample"],
        downsample_type=model_cfg["downsample_type"],
        enc_attn_patterns=enc_attn_patterns,
        dec_self_attn_patterns=dec_self_attn_patterns,
        dec_cross_attn_patterns=dec_cross_attn_patterns,
        dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
        dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
        num_heads=model_cfg["num_heads"],
        attn_drop=model_cfg["attn_drop"],
        proj_drop=model_cfg["proj_drop"],
        ffn_drop=model_cfg["ffn_drop"],
        upsample_type=model_cfg["upsample_type"],
        ffn_activation=model_cfg["ffn_activation"],
        gated_ffn=model_cfg["gated_ffn"],
        norm_layer=model_cfg["norm_layer"],
        num_global_vectors=model_cfg["num_global_vectors"],
        use_dec_self_global=model_cfg["use_dec_self_global"],
        dec_self_update_global=model_cfg["dec_self_update_global"],
        use_dec_cross_global=model_cfg["use_dec_cross_global"],
        use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
        use_global_self_attn=model_cfg["use_global_self_attn"],
        separate_global_qkv=model_cfg["separate_global_qkv"],
        global_dim_ratio=model_cfg["global_dim_ratio"],
        initial_downsample_type=model_cfg["initial_downsample_type"],
        initial_downsample_activation=model_cfg["initial_downsample_activation"],
        initial_downsample_scale=model_cfg["initial_downsample_scale"],
        initial_downsample_conv_layers=model_cfg["initial_downsample_conv_layers"],
        final_upsample_conv_layers=model_cfg["final_upsample_conv_layers"],
        padding_type=model_cfg["padding_type"],
        z_init_method=model_cfg["z_init_method"],
        checkpoint_level=model_cfg["checkpoint_level"],
        pos_embed_type=model_cfg["pos_embed_type"],
        use_relative_pos=model_cfg["use_relative_pos"],
        self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
        attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
        ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
        conv_init_mode=model_cfg["conv_init_mode"],
        down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
        norm_init_mode=model_cfg["norm_init_mode"],
    )


class EarthformerLit(pl.LightningModule):
    """包装 CuboidTransformerModel；batch 为 (seqs_x, seqs_y)，形状 (B,T,C,H,W)。"""

    def __init__(
        self,
        height,
        width,
        input_length,
        target_length,
        downscale_factor,
        learning_rate,
        loss_fx,
        input_class,
        predict_class,
        predict_class_vmax,
        add_video,
        weights_prec,
        thresholds_prec,
        weights_radar,
        thresholds_radar,
        visual_prec_vmin,
        visual_prec_vmax,
        visual_train_steps,
        visual_val_steps,
        train_log_steps,
        val_log_steps,
        test_save_path,
        batch_size,
        earthformer_oc_file="",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs"])

        c_in = len(input_class)
        c_out = len(predict_class)
        assert c_in == c_out, "EarthformerLit 当前按 SimVP 假设输入输出通道数一致"

        cfg_path = earthformer_oc_file or str(_DEFAULT_CFG)
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"Earthformer 配置不存在: {cfg_path}")
        oc = OmegaConf.load(cfg_path)
        model_cfg = OmegaConf.to_object(oc.model)
        model_cfg["input_shape"] = [input_length, height, width, c_in]
        model_cfg["target_shape"] = [target_length, height, width, c_out]

        self.net = _build_cuboid_from_model_cfg(model_cfg)
        self.loss_fx = get_loss_fx(
            loss_fx,
            predict_class,
            predict_class_vmax,
            weights_prec,
            thresholds_prec,
            weights_radar,
            thresholds_radar,
        )

    def _btc_hw_to_bthwc(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 1, 3, 4, 2).contiguous()

    def _bthwc_to_btc_hw(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 1, 4, 2, 3).contiguous()

    def forward(self, seqs_x: torch.Tensor) -> torch.Tensor:
        x = self._btc_hw_to_bthwc(seqs_x)
        y = self.net(x)
        return self._bthwc_to_btc_hw(y)

    def training_step(self, batch, batch_idx):
        seqs_x, seqs_y = batch
        pred = self(seqs_x)
        loss = self.loss_fx(pred, seqs_y)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seqs_x, seqs_y = batch
        pred = self(seqs_x)
        pred_c = torch.clip(pred, 0, 1)
        loss_fx_val = self.loss_fx(pred_c, seqs_y)
        mse = F.mse_loss(pred_c, seqs_y)
        self.log("val_loss", mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid_loss_fx", loss_fx_val, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        seqs_x, seqs_y = batch
        pred = torch.clip(self(seqs_x), 0, 1)
        loss_fx_val = self.loss_fx(pred, seqs_y)
        self.log("test_loss_fx", loss_fx_val, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5)
        return opt

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("EarthformerLit")
        add_parser(parser)
        parser.add_argument(
            "--earthformer_oc_file",
            type=str,
            default="",
            help="Earthformer 模型 yaml（含 model 段）；空则使用 models/earthformer_default_mmnist.yaml",
        )
        return parent_parser
