# 单站雷达 PNG 序列：SingleRadarDataset / SingleRadarDataModule。
# 支持：多列 CSV + PNG 逐帧读；单列逐样本条带 PNG（packed_sequence_file）；
#   事件索引 + 帧目录 .npy（preprocess/huadong_cr_6min/05_export_event_frames_npy.py，dBZ/data_dir_*/frame_*.npy）。
# 读入后按与模型一致的 height/width（如 560×560）在下方、右侧零填充。
# 用法：train.py --dataset_name SingleRadarDataModule；
#   条带模式 --packed_sequence_file 1；事件帧目录 --packed_event_npy 1，radar_dir 指向各 split 根（含 dBZ/ 与 events_manifest.json）。
#   验证子采样：val_sample_interval>1 时验证集仅索引 0, N, 2N, ...。

import csv
import json
import os
import imageio
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset
##################################################### DataSet ##########################################################


####################################### Nowcasting dataset #############################

class SingleRadarDataset(Dataset):

    def __init__(self, csv_file, radar_dir, input_length=20, target_length=20,
                 height=None, width=None, packed_sequence_file=False,
                 packed_event_npy=False):
        """
        :param csv_file: PNG 模式：每行逗号分隔相对路径，总列数 = input_length+target_length。
            条带模式：每行一个条带 .png 相对路径（相对 radar_dir）。
            事件帧模式：表头 event_id,start_frame,start_time；自 radar_dir/dBZ/data_dir_{eid:03d}/frame_{i:05d}.npy 读 T 帧。
        :param radar_dir: 数据根目录；PNG 模式与 preprocess ROOT_DIR 一致；逐样本条带模式为条带根目录；
            事件帧模式为单 split 目录（如 .../event_npy/train），其下含 dBZ/ 与 events_manifest.json。
        :param input_length: 输入序列长度（须与 train.py / 预处理 INPUT_FRAMES 一致）
        :param target_length: 预报序列长度（须与 TARGET_FRAMES 一致）
        :param height: 与模型 --height 一致；与 width 同时给出时，将每帧补零至 (height, width)。均为 None 则不填充。
        :param width: 与模型 --width 一致。
        :param packed_sequence_file: True 时单列条带 PNG 路径，形状 (H,T*W)，每行一个相对路径。
        :param packed_event_npy: True 时 CSV 为 05_export_event_frames_npy 生成的事件索引；与 packed_sequence_file 互斥。
        """
        super(SingleRadarDataset, self).__init__()
        self._event_npy = bool(packed_event_npy)
        self._packed = bool(packed_sequence_file)
        self._event_manifest = None
        if self._event_npy and self._packed:
            raise ValueError("packed_event_npy 与 packed_sequence_file 不能同时为 True。")
        if self._event_npy:
            self._samples = []
            with open(csv_file, encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    self._samples.append(
                        (int(row["event_id"]), int(row["start_frame"]))
                    )
            self.fname = None
            man_path = os.path.join(radar_dir, "events_manifest.json")
            if os.path.isfile(man_path):
                with open(man_path, encoding="utf-8") as fp:
                    self._event_manifest = json.load(fp)
            else:
                self._event_manifest = None
        elif self._packed:
            with open(csv_file, encoding="utf-8") as fp:
                lines = [ln.strip() for ln in fp if ln.strip()]
            self.fname = np.array(lines, dtype=str).reshape(-1, 1)
        else:
            self.fname = np.loadtxt(csv_file, delimiter=",", dtype=str)
            if self.fname.ndim == 1:
                self.fname = self.fname[np.newaxis, :]
        self.radar_dir = radar_dir
        self.input_length = int(input_length)
        self.target_length = int(target_length)
        self._total_frames = self.input_length + self.target_length
        if (height is None) ^ (width is None):
            raise ValueError("height 与 width 须同时指定或同时为 None。")
        self._target_hw = None
        if height is not None and width is not None:
            self._target_hw = (int(height), int(width))

    def _pad_bottom_right(self, data_radar):
        """(T,H,W) 仅在下方、右侧补零到 _target_hw。"""
        if self._target_hw is None:
            return data_radar
        th, tw = self._target_hw
        _, h, w = data_radar.shape
        pad_bottom = th - h
        pad_right = tw - w
        if pad_bottom < 0 or pad_right < 0:
            raise ValueError(
                f"目标 (height,width)=({th},{tw}) 小于原始尺寸 ({h},{w})，无法仅在下/右填充。"
            )
        if pad_bottom == 0 and pad_right == 0:
            return data_radar
        return np.pad(
            data_radar,
            ((0, 0), (0, pad_bottom), (0, pad_right)),
            mode="constant",
            constant_values=0.0,
        )

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        if self._event_npy:
            eid, sf = self._samples[item]
            ev_dir = os.path.join(self.radar_dir, "dBZ", f"data_dir_{eid:03d}")
            t_need = self._total_frames
            if self._event_manifest is not None:
                n_all = int(self._event_manifest[str(eid)]["n_frames"])
            else:
                n_all = None
            if n_all is not None and sf + t_need > n_all:
                raise ValueError(
                    f"事件 {eid} 总帧数={n_all}，无法从 start_frame={sf} 取 {t_need} 帧: {ev_dir}"
                )
            frames = []
            for i in range(sf, sf + t_need):
                fp = os.path.join(ev_dir, f"frame_{i:05d}.npy")
                arr = np.load(fp)
                if arr.ndim == 3:
                    arr = arr[..., 0]
                frames.append(arr.astype(np.float32, copy=False))
            data_radar = np.stack(frames, axis=0) / 255.0
        elif self._packed:
            paths = self.fname[item]
            if paths.size != 1:
                raise ValueError("条带模式 CSV 每行应仅一个 .png 相对路径")
            rel = paths[0]
            fpath = os.path.join(self.radar_dir, rel)
            ext = os.path.splitext(rel)[1].lower()
            if ext != ".png":
                raise ValueError(
                    f"条带模式仅支持横向条带 PNG；当前后缀: {ext!r}"
                )
            strip = np.asarray(imageio.imread(fpath), dtype=np.float32)
            if strip.ndim == 3:
                strip = strip[..., 0]
            t = self._total_frames
            h, tw = strip.shape
            if tw % t != 0:
                raise ValueError(
                    f"条带 PNG 宽度 {tw} 不能被帧数 {t} 整除: {fpath}"
                )
            w = tw // t
            data_radar = strip.reshape(h, t, w).transpose(1, 0, 2) / 255.0
        else:
            paths = self.fname[item]
            if paths.size != self._total_frames:
                raise ValueError(
                    f"CSV 第 {item} 行列数={paths.size}，与 input_length+target_length={self._total_frames} 不符"
                )
            data_radar = np.asarray(
                [imageio.imread(os.path.join(self.radar_dir, ifilename)) for ifilename in paths],
                dtype=np.float32,
            ) / 255.0

        if data_radar.shape[0] != self._total_frames:
            raise ValueError(
                f"序列长度 {data_radar.shape[0]} != input_length+target_length={self._total_frames}"
            )

        data_radar = self._pad_bottom_right(data_radar)

        t_in = self.input_length
        t_out = self.target_length
        return (
            data_radar[:t_in, np.newaxis, ...],
            data_radar[t_in : t_in + t_out, np.newaxis, ...],
        )

    def __len__(self):
        if self._event_npy:
            return len(self._samples)
        return self.fname.shape[0]


class SingleRadarDataModule(pl.LightningDataModule):

    def __init__(self, train_file="D:\Code\Tianchi\data\dataset_train.csv",
                 val_file="D:\Code\Tianchi\data\dataset_testA.csv",
                 test_file="D:\Code\Tianchi\data\dataset_testA.csv",
                 radar_dir="D:\Code\Tianchi\data\Train\Radar",
                 radar_dir_val=None,
                 radar_dir_test=None,
                 input_length=20, target_length=20,
                 height=None, width=None,
                 packed_sequence_file=False,
                 packed_event_npy=False,
                 val_sample_interval=1,
                 num_workers=8, batch_size=10, pin_memory=True,
                 prefetch_factor=4, **kwargs):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = int(prefetch_factor) if prefetch_factor is not None else 4
        self.radar_dir = radar_dir
        self.radar_dir_val = radar_dir if radar_dir_val is None else radar_dir_val
        self.radar_dir_test = radar_dir if radar_dir_test is None else radar_dir_test
        self.input_length = int(input_length)
        self.target_length = int(target_length)
        self.height = height
        self.width = width
        self.packed_sequence_file = bool(packed_sequence_file)
        self.packed_event_npy = bool(packed_event_npy)
        self.val_sample_interval = max(1, int(val_sample_interval))

    def _dataloader(self, dataset, shuffle):
        nw = self.num_workers
        kw = dict(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=bool(self.pin_memory),
        )
        if nw > 0:
            kw["persistent_workers"] = True
            kw["prefetch_factor"] = max(2, self.prefetch_factor)
        return DataLoader(**kw)

    def setup(self, stage=None):
        kw = dict(
            input_length=self.input_length,
            target_length=self.target_length,
            height=self.height,
            width=self.width,
            packed_sequence_file=self.packed_sequence_file,
            packed_event_npy=self.packed_event_npy,
        )
        self.train = SingleRadarDataset(self.train_file, self.radar_dir, **kw)
        self.val = SingleRadarDataset(self.val_file, self.radar_dir_val, **kw)
        if self.val_sample_interval > 1:
            n = len(self.val)
            idx = list(range(0, n, self.val_sample_interval))
            self.val = Subset(self.val, idx)
        self.test = SingleRadarDataset(self.test_file, self.radar_dir_test, **kw)

    def train_dataloader(self):
        return self._dataloader(self.train, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val, shuffle=False)

    def test_dataloader(self):
        return self._dataloader(self.test, shuffle=False)
