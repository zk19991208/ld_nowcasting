from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_simvp_nc import (
    DEFAULT_BACKGROUND,
    DEFAULT_CITY_SHP,
    DEFAULT_EXTENT,
    DEFAULT_PROVINCE_SHP,
    LEVELS,
    draw_radar,
    load_map_resources,
    setup_map,
)


DEFAULT_START_TIME = "202508240500"
DEFAULT_END_TIME = "202508241500"
DEFAULT_RADAR_DIR = "data"
INTERVAL_MINUTES = 6


def parse_args(default_extent=DEFAULT_EXTENT) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot raw Xinjiang radar PNG maps every 6 minutes")
    parser.add_argument("--start_time", default=DEFAULT_START_TIME, help="YYYYmmddHHMM, inclusive.")
    parser.add_argument("--end_time", default=DEFAULT_END_TIME, help="YYYYmmddHHMM, inclusive.")
    parser.add_argument(
        "--radar_dir",
        default=DEFAULT_RADAR_DIR,
        help="Data root or the xinjiang/CR_6min_550x550 directory.",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--province_shp", default=DEFAULT_PROVINCE_SHP)
    parser.add_argument("--city_shp", default=DEFAULT_CITY_SHP)
    parser.add_argument("--background", default=DEFAULT_BACKGROUND)
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        default=default_extent,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
    )
    return parser.parse_args()


def iter_times(start_time: datetime, end_time: datetime):
    if end_time < start_time:
        raise ValueError("end_time must not be earlier than start_time")
    current = start_time
    while current <= end_time:
        yield current
        current += timedelta(minutes=INTERVAL_MINUTES)


def radar_path(root: Path, valid_time: datetime) -> Path:
    filename = valid_time.strftime("%Y%m%d%H%M") + ".png"
    relative = Path(valid_time.strftime("%Y")) / valid_time.strftime("%Y%m%d") / (
        filename
    )
    candidates = [
        root / filename,
        root / relative,
        root / "xinjiang" / "CR_6min_550x550" / relative,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Radar image not found. Tried:\n  " + "\n  ".join(str(path) for path in candidates)
    )


def read_dbz(path: Path) -> np.ndarray:
    image = np.asarray(plt.imread(path))
    if image.ndim == 3:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D radar image, got {image.shape}: {path}")
    image = image.astype(np.float32)
    if image.max(initial=0.0) > 1.0:
        image /= 255.0
    return image * 70.0


def save_map(
    dbz,
    output_path,
    projection,
    provinces,
    cities,
    background_image,
    extent,
):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    setup_map(ax, projection, provinces, cities, background_image, extent, labels=True)
    image = draw_radar(ax, dbz, projection, extent)
    colorbar = fig.colorbar(image, ax=ax, ticks=LEVELS[1:], shrink=0.82, pad=0.04)
    colorbar.set_label("Reflectivity (dBZ)")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(default_extent=DEFAULT_EXTENT, output_prefix="original") -> None:
    args = parse_args(default_extent)
    start_time = datetime.strptime(args.start_time, "%Y%m%d%H%M")
    end_time = datetime.strptime(args.end_time, "%Y%m%d%H%M")
    radar_dir = Path(args.radar_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("plot/output") / (
        f"{output_prefix}_{args.start_time}_{args.end_time}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_times = list(iter_times(start_time, end_time))
    paths = [radar_path(radar_dir, valid_time) for valid_time in valid_times]
    projection, provinces, cities, background_image = load_map_resources(
        Path(args.province_shp), Path(args.city_shp), Path(args.background)
    )

    for index, (valid_time, path) in enumerate(zip(valid_times, paths), start=1):
        output_path = output_dir / f"original_{valid_time:%Y%m%d%H%M}.png"
        save_map(
            read_dbz(path),
            output_path,
            projection,
            provinces,
            cities,
            background_image,
            args.extent,
        )
        print(f"[{index:03d}/{len(valid_times):03d}] {output_path}")

    print(f"Saved {len(valid_times)} original radar maps: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
