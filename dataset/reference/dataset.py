"""Radar nowcast dataset and Lightning DataModule.

Loads samples via pre-built index CSV files (see build_index.py).
Supports single variable (dBZ) and multi-variable (dBZ + ZDR + KDP).
"""

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L


# Per-variable normalization: clip range then scale to [0, 1]
NORM_PARAMS = {
    "dBZ": {"vmin": 0.0, "vmax": 70.0},
    "ZDR": {"vmin": -5.0, "vmax": 5.0},
    "KDP": {"vmin": -2.0, "vmax": 6.0},
}


def _load_index(csv_path: str) -> List[Tuple[int, int]]:
    """Load (event_id, start_frame) pairs from a CSV file."""
    samples = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append((int(row["event_id"]), int(row["start_frame"])))
    return samples


def _normalize(data: np.ndarray, variable: str) -> np.ndarray:
    p = NORM_PARAMS[variable]
    data = np.clip(data, p["vmin"], p["vmax"])
    return (data - p["vmin"]) / (p["vmax"] - p["vmin"])


class RadarDataset(Dataset):
    """Radar echo extrapolation dataset.

    Each sample consists of:
        input:  (n_input * n_vars, H, W)  -- past frames
        target: (n_output * n_vars, H, W) -- future frames
    """

    def __init__(
        self,
        data_root: str,
        index_file: str,
        variables: List[str],
        n_input: int,
        n_output: int,
    ):
        self.data_root = data_root
        self.variables = variables
        self.n_input = n_input
        self.n_output = n_output
        self.window = n_input + n_output
        self.samples = _load_index(index_file)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frames(self, variable: str, event_id: int, start: int) -> np.ndarray:
        """Load a sequence of frames for one variable. Returns (T, H, W)."""
        var_dir = os.path.join(
            self.data_root, variable, f"data_dir_{event_id:03d}"
        )
        frames = []
        for i in range(start, start + self.window):
            path = os.path.join(var_dir, f"frame_{i:03d}.npy")
            frame = np.load(path).astype(np.float32)
            frame = _normalize(frame, variable)
            frames.append(frame)
        return np.stack(frames, axis=0)  # (T, H, W)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        event_id, start_frame = self.samples[idx]

        var_sequences = []
        for var in self.variables:
            seq = self._load_frames(var, event_id, start_frame)  # (T, H, W)
            var_sequences.append(seq)

        # (T, C, H, W) where C = number of variables
        data = np.stack(var_sequences, axis=1)

        # Flatten time and channel: (T, C, H, W) -> (T*C, H, W)
        T, C, H, W = data.shape
        inp = data[: self.n_input].reshape(self.n_input * C, H, W)
        tgt = data[self.n_input :].reshape(self.n_output * C, H, W)

        return torch.from_numpy(inp), torch.from_numpy(tgt)


class RadarDataModule(L.LightningDataModule):
    """Lightning DataModule for radar nowcast training."""

    def __init__(self, cfg: Dict):
        super().__init__()
        data_cfg = cfg["data"]
        self.data_root = data_cfg["data_root"]
        self.index_dir = data_cfg["index_dir"]
        self.variables = data_cfg["variables"]
        self.n_input = data_cfg["n_input_frames"]
        self.n_output = data_cfg["n_output_frames"]
        self.batch_size = data_cfg["batch_size"]
        self.num_workers = data_cfg["num_workers"]

    def _make_dataset(self, split: str) -> RadarDataset:
        index_file = os.path.join(self.index_dir, f"{split}_index.csv")
        return RadarDataset(
            data_root=self.data_root,
            index_file=index_file,
            variables=self.variables,
            n_input=self.n_input,
            n_output=self.n_output,
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_set = self._make_dataset("train")
            self.val_set = self._make_dataset("val")
        if stage == "test" or stage is None:
            self.test_set = self._make_dataset("test")

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
