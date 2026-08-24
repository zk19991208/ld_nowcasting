from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


TRANSFER_ROOT = Path(__file__).resolve().parent.parent
if str(TRANSFER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANSFER_ROOT))

from models import SimVP_Lit


DEFAULT_WEIGHT = (
    "save/simvp_mixed_cr550_png_huadong/"
    "weights-epoch=006-CSI_35dBZ_val=0.070.ckpt"
)
DEFAULT_START_TIME = "202508240100"
DEFAULT_RADAR_DIR = "data"
INTERVAL_MINUTES = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimVP inference from historical radar PNG files")
    parser.add_argument("--weight_path", default=DEFAULT_WEIGHT)
    parser.add_argument(
        "--start_time",
        default=DEFAULT_START_TIME,
        help="Forecast reference time in YYYYmmddHHMM; this is the last input frame.",
    )
    parser.add_argument(
        "--radar_dir",
        default=DEFAULT_RADAR_DIR,
        help="Data root or the xinjiang/CR_6min_550x550 directory.",
    )
    parser.add_argument("--output", default=None, help="Output .nc path.")
    parser.add_argument("--device", default="cuda:0", help="For example cuda:0, npu:0, or cpu.")
    return parser.parse_args()


def radar_path(root: Path, valid_time: datetime) -> Path:
    relative = Path(valid_time.strftime("%Y")) / valid_time.strftime("%Y%m%d") / (
        valid_time.strftime("%Y%m%d%H%M") + ".png"
    )
    candidates = [
        root / relative,
        root / "xinjiang" / "CR_6min_550x550" / relative,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Radar image not found. Tried:\n  " + "\n  ".join(str(path) for path in candidates)
    )


def load_history(
    radar_dir: Path,
    start_time: datetime,
    input_length: int,
    model_height: int,
    model_width: int,
) -> tuple[torch.Tensor, list[datetime], tuple[int, int], list[Path]]:
    times = [
        start_time - timedelta(minutes=INTERVAL_MINUTES * i)
        for i in range(input_length - 1, -1, -1)
    ]
    paths = [radar_path(radar_dir, valid_time) for valid_time in times]
    frames = []
    original_shape = None

    for path in paths:
        frame = np.asarray(imageio.imread(path))
        if frame.ndim == 3:
            frame = frame[..., 0]
        if frame.ndim != 2:
            raise ValueError(f"Expected a 2-D radar image, got {frame.shape}: {path}")
        if original_shape is None:
            original_shape = frame.shape
        elif frame.shape != original_shape:
            raise ValueError(f"Inconsistent image shape {frame.shape}, expected {original_shape}: {path}")
        frames.append(frame.astype(np.float32) / 255.0)

    assert original_shape is not None
    raw_height, raw_width = original_shape
    if raw_height > model_height or raw_width > model_width:
        raise ValueError(
            f"Input image {original_shape} is larger than model shape "
            f"({model_height}, {model_width})"
        )

    array = np.stack(frames, axis=0)
    array = np.pad(
        array,
        ((0, 0), (0, model_height - raw_height), (0, model_width - raw_width)),
        mode="constant",
    )
    # Model input: (batch, time, channel, height, width).
    tensor = torch.from_numpy(array[:, None, :, :]).unsqueeze(0)
    return tensor, times, original_shape, paths


def save_netcdf(
    output_path: Path,
    forecast_dbz: np.ndarray,
    forecast_times: list[datetime],
    start_time: datetime,
    weight_path: Path,
    input_paths: list[Path],
) -> None:
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise RuntimeError("Saving .nc requires netCDF4: pip install netCDF4") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epoch = datetime(1970, 1, 1)
    seconds = np.asarray([(t - epoch).total_seconds() for t in forecast_times], dtype=np.int64)
    leads = np.asarray(
        [int((t - start_time).total_seconds() // 60) for t in forecast_times],
        dtype=np.int32,
    )

    with Dataset(output_path, "w", format="NETCDF4") as nc:
        nc.createDimension("time", len(forecast_times))
        nc.createDimension("y", forecast_dbz.shape[1])
        nc.createDimension("x", forecast_dbz.shape[2])

        time_var = nc.createVariable("time", "i8", ("time",))
        time_var[:] = seconds
        time_var.units = "seconds since 1970-01-01 00:00:00"
        time_var.calendar = "standard"

        lead_var = nc.createVariable("lead_time", "i4", ("time",))
        lead_var[:] = leads
        lead_var.units = "minutes"

        dbz_var = nc.createVariable(
            "forecast_dbz",
            "f4",
            ("time", "y", "x"),
            zlib=True,
            complevel=4,
        )
        dbz_var[:] = forecast_dbz
        dbz_var.long_name = "SimVP forecast radar reflectivity"
        dbz_var.units = "dBZ"
        dbz_var.valid_min = np.float32(0.0)
        dbz_var.valid_max = np.float32(70.0)

        nc.title = "SimVP radar extrapolation"
        nc.forecast_reference_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
        nc.model_checkpoint = str(weight_path.resolve())
        nc.input_files = ",".join(str(path) for path in input_paths)


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("medium")

    start_time = datetime.strptime(args.start_time, "%Y%m%d%H%M")
    weight_path = Path(args.weight_path)
    radar_dir = Path(args.radar_dir)
    output_path = (
        Path(args.output)
        if args.output
        else Path("inference/output") / f"simvp_{args.start_time}.nc"
    )

    if not weight_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weight_path}")

    if args.device.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Using an NPU requires torch_npu") from exc
    device = torch.device(args.device)

    model = SimVP_Lit.load_from_checkpoint(
        str(weight_path),
        strict=False,
        map_location="cpu",
        batch_size=1,
        test_save_path="",
    )
    model.freeze()
    model.eval()
    model.to(device)

    hp = model.hparams
    history, history_times, original_shape, input_paths = load_history(
        radar_dir=radar_dir,
        start_time=start_time,
        input_length=int(hp.input_length),
        model_height=int(hp.height),
        model_width=int(hp.width),
    )

    with torch.inference_mode():
        prediction = torch.clamp(model(history.to(device)), 0.0, 1.0)

    raw_height, raw_width = original_shape
    predict_classes = list(hp.predict_class)
    radar_channel = predict_classes.index(1) if 1 in predict_classes else 0
    vmax = float(list(hp.predict_class_vmax)[radar_channel])
    forecast_dbz = (
        prediction[0, :, radar_channel, :raw_height, :raw_width].float().cpu().numpy()
        * vmax
    )
    forecast_times = [
        start_time + timedelta(minutes=INTERVAL_MINUTES * i)
        for i in range(1, forecast_dbz.shape[0] + 1)
    ]

    save_netcdf(
        output_path=output_path,
        forecast_dbz=forecast_dbz,
        forecast_times=forecast_times,
        start_time=start_time,
        weight_path=weight_path,
        input_paths=input_paths,
    )

    print(f"Input: {history_times[0]:%Y%m%d%H%M} - {history_times[-1]:%Y%m%d%H%M}")
    print(f"Forecast: {forecast_times[0]:%Y%m%d%H%M} - {forecast_times[-1]:%Y%m%d%H%M}")
    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
