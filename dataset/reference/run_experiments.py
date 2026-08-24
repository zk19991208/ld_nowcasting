#!/usr/bin/env python
"""Run a suite of experiments comparing loss functions and model modules.

Each experiment runs sequentially using all available GPUs (DDP).
Results are saved per-experiment and compiled into a summary table.

Usage:
    python run_experiments.py                    # run all experiments
    python run_experiments.py --only A,B,G       # run selected experiments
    python run_experiments.py --summary-only     # just rebuild summary from existing results
"""

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("experiments")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.join(BASE_DIR, "experiments")
BASE_CFG_PATH = os.path.join(BASE_DIR, "config.yaml")

EXPERIMENTS = {
    "A": {
        "desc": "MSE (baseline)",
        "loss": {"components": {"mse": {"weight": 1.0}}},
        "mefm": False,
    },
    "B": {
        "desc": "MSE + PM(ω=10) — Cao 2025",
        "loss": {"components": {"mse": {"weight": 1.0}, "pm": {"weight": 10.0}}},
        "mefm": False,
    },
    "C": {
        "desc": "Weighted MSE (exp=2)",
        "loss": {"components": {"weighted_mse": {"weight": 1.0, "exponent": 2.0}}},
        "mefm": False,
    },
    "D": {
        "desc": "Balanced MSE + Balanced MAE",
        "loss": {"components": {
            "balanced_mse": {"weight": 1.0},
            "balanced_mae": {"weight": 1.0},
        }},
        "mefm": False,
    },
    "E": {
        "desc": "MSE + SSIM(w=0.5)",
        "loss": {"components": {"mse": {"weight": 1.0}, "ssim": {"weight": 0.5}}},
        "mefm": False,
    },
    "F": {
        "desc": "CM loss (Yang & Yuan 2023, fixed)",
        "loss": {"components": {
            "balanced_mse": {"weight": 1.0},
            "balanced_mae": {"weight": 1.0},
            "spatial_ms_ssim": {"weight": 1.0, "n_levels": 3},
            "temporal": {"weight": 1.0, "n_output_frames": 20, "n_vars": 1},
        }},
        "mefm": False,
    },
    "G": {
        "desc": "MSE + PM(ω=10) + MEFM",
        "loss": {"components": {"mse": {"weight": 1.0}, "pm": {"weight": 10.0}}},
        "mefm": True,
    },
    "H": {
        "desc": "CM loss + MEFM (full CM framework)",
        "loss": {"components": {
            "balanced_mse": {"weight": 1.0},
            "balanced_mae": {"weight": 1.0},
            "spatial_ms_ssim": {"weight": 1.0, "n_levels": 3},
            "temporal": {"weight": 1.0, "n_output_frames": 20, "n_vars": 1},
        }},
        "mefm": True,
    },
    "I": {
        "desc": "MSE + Dice(t=0.286)",
        "loss": {"components": {
            "mse": {"weight": 1.0},
            "dice": {"weight": 1.0, "threshold": 0.286},
        }},
        "mefm": False,
    },
    "J": {
        "desc": "WADEPre + MSE",
        "loss": {"components": {"mse": {"weight": 1.0}}},
        "mefm": False,
        "model_type": "wadepre",
    },
    "K": {
        "desc": "AlphaPre + MSE",
        "loss": {"components": {"mse": {"weight": 1.0}}},
        "mefm": False,
        "model_type": "alphapre",
    },
    "L": {
        "desc": "DiffCast (SimVP + Residual Diffusion)",
        "loss": {"use_diffcast_loss": True, "components": {"mse": {"weight": 1.0}}},
        "mefm": False,
        "model_type": "diffcast",
    },
}


def build_config(base_cfg: dict, exp_def: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["loss"] = exp_def["loss"]
    cfg["model"]["mefm"]["enabled"] = exp_def["mefm"]

    model_type = exp_def.get("model_type", "unet")
    cfg["model"]["type"] = model_type
    if model_type == "wadepre":
        cfg["model"]["wadepre"] = {
            "spatial_size": 256,
            "hidden_size": 128,
            "wavelet_level": 3,
            "refine_hidden": 120,
        }
        cfg["data"]["batch_size"] = 8
    elif model_type == "alphapre":
        cfg["model"]["alphapre"] = {
            "spatial_size": 256,
            "hidden_dim": 64,
            "n_layers": 3,
            "spec_num": 20,
        }
        cfg["data"]["batch_size"] = 2
    elif model_type == "diffcast":
        cfg["model"]["diffcast"] = {
            "spatial_size": 256,
            "dim": 64,
            "dim_mults": [1, 2, 4, 8],
            "diffusion_timesteps": 1000,
            "sampling_timesteps": 250,
            "objective": "pred_v",
            "loss_alpha": 0.5,
            "simvp_hid_S": 64,
            "simvp_hid_T": 256,
            "simvp_N_S": 2,
            "simvp_N_T": 6,
        }
        cfg["data"]["batch_size"] = 1
        cfg["train"]["lr"] = 0.00005
        cfg["train"]["weight_decay"] = 0.0
        cfg["train"]["optimizer"] = "adamw"
        cfg["train"]["scheduler"] = {"type": "cosine"}
        cfg["train"]["early_stopping_patience"] = 50
        cfg["train"]["gradient_clip_val"] = 1.0
        cfg["train"]["precision"] = "bf16-mixed"

    cfg["train"]["max_epochs"] = 500
    if model_type != "diffcast":
        cfg["train"]["early_stopping_patience"] = 10
    cfg["vis"]["enabled"] = True
    cfg["vis"]["every_n_epochs"] = 5
    cfg["vis"]["val_every_n_steps"] = 0
    cfg["vis"]["train_every_n_steps"] = 0
    cfg["vis"]["save_dir"] = "./vis_output"
    return cfg


def run_one(exp_id: str, exp_def: dict, base_cfg: dict) -> dict:
    exp_dir = os.path.join(EXP_ROOT, f"EXP_{exp_id}")
    os.makedirs(exp_dir, exist_ok=True)

    cfg = build_config(base_cfg, exp_def)
    cfg_path = os.path.join(exp_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    log.info("=" * 70)
    log.info("  EXP_%s: %s", exp_id, exp_def["desc"])
    log.info("  Model: %s | MEFM: %s | Loss: %s",
             exp_def.get("model_type", "unet"), exp_def["mefm"],
             list(exp_def["loss"]["components"].keys()))
    log.info("  Output: %s", exp_dir)
    log.info("=" * 70)

    t0 = time.time()
    log_path = os.path.join(exp_dir, "train.log")

    cmd = [
        sys.executable, os.path.join(BASE_DIR, "train.py"),
        "--config", cfg_path,
        "--output-dir", exp_dir,
    ]

    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                              cwd=BASE_DIR)

    elapsed = time.time() - t0
    log.info("  EXP_%s finished in %s  (exit=%d)",
             exp_id, str(timedelta(seconds=int(elapsed))), proc.returncode)

    result = {"id": exp_id, "desc": exp_def["desc"],
              "mefm": exp_def["mefm"], "elapsed_s": int(elapsed),
              "exit_code": proc.returncode}

    res_path = os.path.join(exp_dir, "results.json")
    if os.path.exists(res_path):
        with open(res_path) as f:
            result.update(json.load(f))
    else:
        log.warning("  results.json not found for EXP_%s", exp_id)

    return result


def collect_results(exp_ids):
    rows = []
    for eid in exp_ids:
        rp = os.path.join(EXP_ROOT, f"EXP_{eid}", "results.json")
        if not os.path.exists(rp):
            continue
        with open(rp) as f:
            r = json.load(f)
        r["id"] = eid
        r["desc"] = EXPERIMENTS[eid]["desc"]
        r["mefm"] = EXPERIMENTS[eid]["mefm"]
        rows.append(r)
    return rows


def write_summary(results):
    os.makedirs(EXP_ROOT, exist_ok=True)
    md_path = os.path.join(EXP_ROOT, "summary.md")
    json_path = os.path.join(EXP_ROOT, "summary.json")

    metrics_keys = [
        "test/CSI_20dBZ", "test/CSI_35dBZ", "test/CSI_40dBZ",
        "test/POD_20dBZ", "test/POD_35dBZ", "test/POD_40dBZ",
        "test/FAR_20dBZ", "test/FAR_35dBZ", "test/FAR_40dBZ",
    ]

    header = ("| ID | 实验名称 | Model | MEFM | Epochs | Best val_loss "
              "| CSI_20 | CSI_35 | CSI_40 "
              "| POD_20 | POD_35 | POD_40 "
              "| FAR_20 | FAR_35 | FAR_40 |")
    sep = "|" + "|".join(["---"] * 15) + "|"

    lines = [
        f"# 实验结果汇总  ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n",
        header, sep,
    ]

    best_csi20 = (-1, "")
    best_val = (1e9, "")

    for r in results:
        tm = r.get("test_metrics", {})
        eid = r.get("id", "?")
        desc = r.get("desc", "")
        model_type = EXPERIMENTS.get(eid, {}).get("model_type", "unet")
        mefm = "✓" if r.get("mefm") else "✗"
        epochs = r.get("epochs_trained", "?")
        vloss = r.get("best_val_loss")
        vloss_s = f"{vloss:.6f}" if vloss is not None else "N/A"

        vals = []
        for mk in metrics_keys:
            v = tm.get(mk)
            vals.append(f"{v:.4f}" if v is not None else "N/A")

        line = f"| {eid} | {desc} | {model_type} | {mefm} | {epochs} | {vloss_s} | " + " | ".join(vals) + " |"
        lines.append(line)

        csi20 = tm.get("test/CSI_20dBZ")
        if csi20 is not None and csi20 > best_csi20[0]:
            best_csi20 = (csi20, eid)
        if vloss is not None and vloss < best_val[0]:
            best_val = (vloss, eid)

    lines.append("")
    lines.append(f"**最佳 val_loss**: EXP_{best_val[1]} ({best_val[0]:.6f})")
    lines.append(f"**最佳 CSI_20dBZ**: EXP_{best_csi20[1]} ({best_csi20[0]:.4f})")
    lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info("Summary written to %s", md_path)
    print("\n" + "\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated experiment IDs to run (e.g. A,B,G)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip training, just rebuild summary from results.json")
    args = parser.parse_args()

    exp_ids = list(EXPERIMENTS.keys())
    if args.only:
        exp_ids = [x.strip().upper() for x in args.only.split(",")]
        for eid in exp_ids:
            if eid not in EXPERIMENTS:
                log.error("Unknown experiment ID: %s", eid)
                sys.exit(1)

    if args.summary_only:
        results = collect_results(exp_ids)
        write_summary(results)
        return

    base_cfg = yaml.safe_load(open(BASE_CFG_PATH))
    log.info("Running %d experiments: %s", len(exp_ids), exp_ids)
    log.info("max_epochs=500, early_stopping_patience=10")

    results = []
    for eid in exp_ids:
        result = run_one(eid, EXPERIMENTS[eid], base_cfg)
        results.append(result)

    write_summary(results)


if __name__ == "__main__":
    main()
