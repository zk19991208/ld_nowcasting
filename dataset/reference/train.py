"""Training entry point for radar nowcast UNet demo.

Usage:
    python build_index.py          # once
    python train.py                # train with config.yaml
    python train.py --config config.yaml --train.max_epochs 100
    python train.py --config exp.yaml --output-dir experiments/run1
"""

import argparse
import json
import logging
import os

import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from dataset import RadarDataModule
from model import RadarNowcastModule
from callbacks import NowcastVisualizationCallback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]):
    """Apply --key.subkey value overrides to the config dict."""
    i = 0
    while i < len(overrides):
        key = overrides[i].lstrip("-")
        val = overrides[i + 1]
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d[p]
        old = d.get(parts[-1])
        if old is not None:
            target_type = type(old)
            if target_type is bool:
                val = val.lower() in ("true", "1", "yes")
            elif target_type is list:
                val = yaml.safe_load(val)
            else:
                val = target_type(val)
        d[parts[-1]] = val
        i += 2


def main():
    parser = argparse.ArgumentParser(description="Radar Nowcast UNet Training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Root directory for logs, checkpoints, vis, results.json")
    parser.add_argument("--ckpt-path", type=str, default=None,
                        help="Resume training from this checkpoint (path to .ckpt file)")
    args, extra = parser.parse_known_args()

    cfg = load_config(args.config)
    if extra:
        apply_overrides(cfg, extra)

    out_dir = args.output_dir or "."
    os.makedirs(out_dir, exist_ok=True)

    L.seed_everything(42, workers=True)

    dm = RadarDataModule(cfg)
    model = RadarNowcastModule(cfg)

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(out_dir, "checkpoints"),
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            filename="epoch{epoch:02d}-val_loss{val_loss:.4f}",
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=cfg["train"].get("early_stopping_patience", 10),
            mode="min",
        ),
    ]

    vis_cfg = cfg.get("vis", {})
    if vis_cfg.get("enabled", False):
        vis_save = vis_cfg.get("save_dir")
        if vis_save and args.output_dir:
            vis_save = os.path.join(out_dir, os.path.basename(vis_save))
        callbacks.append(
            NowcastVisualizationCallback(
                display_lead_minutes=vis_cfg.get("display_lead_minutes",
                                                  [6, 12, 30, 60, 90, 120]),
                every_n_epochs=vis_cfg.get("every_n_epochs", 1),
                val_every_n_steps=vis_cfg.get("val_every_n_steps", 0),
                train_every_n_steps=vis_cfg.get("train_every_n_steps", 0),
                save_dir=vis_save,
            )
        )

    # ── Loggers ──────────────────────────────────────────────────────
    log_cfg = cfg.get("logging", {})
    loggers = []

    tb_dir = os.path.join(out_dir, "lightning_logs")
    if log_cfg.get("use_tensorboard", True):
        loggers.append(TensorBoardLogger(save_dir=tb_dir, name="radar_nowcast"))

    if log_cfg.get("use_wandb", False):
        from lightning.pytorch.loggers import WandbLogger
        loggers.append(
            WandbLogger(
                project=log_cfg.get("wandb_project", "radar_nowcast"),
                entity=log_cfg.get("wandb_entity"),
                log_model=False,
                save_dir=tb_dir,
            )
        )

    if not loggers:
        loggers = True

    train_cfg = cfg["train"]
    trainer = L.Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        logger=loggers,
        precision=train_cfg.get("precision", "16-mixed"),
        gradient_clip_val=train_cfg.get("gradient_clip_val", None),
        log_every_n_steps=20,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)
    test_out = trainer.test(model, datamodule=dm, ckpt_path="best")

    # ── Save results ─────────────────────────────────────────────────
    if trainer.global_rank == 0:
        results = {
            "epochs_trained": trainer.current_epoch + 1,
            "best_val_loss": None,
            "test_metrics": {},
        }
        best_cb = trainer.checkpoint_callback
        if best_cb and best_cb.best_model_score is not None:
            results["best_val_loss"] = best_cb.best_model_score.item()

        if test_out:
            results["test_metrics"] = {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in test_out[0].items()
            }
        for k, v in trainer.callback_metrics.items():
            if k.startswith("test/") or k.startswith("val/"):
                results["test_metrics"][k] = round(v.item(), 6) if hasattr(v, "item") else v

        res_path = os.path.join(out_dir, "results.json")
        with open(res_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", res_path)


if __name__ == "__main__":
    main()
