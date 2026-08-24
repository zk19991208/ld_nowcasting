from __future__ import annotations

import argparse
from pathlib import Path

from plot_original_png import read_dbz, save_map
from plot_simvp_nc import (
    DEFAULT_BACKGROUND,
    DEFAULT_PROVINCE_SHP,
    load_map_resources,
)


DEFAULT_INPUT = "plot/huadong_png/202506011248.png"
DEFAULT_OUTPUT_DIR = "plot/output/huadong_png"
DEFAULT_CITY_SHP = "map/city.shp"
DEFAULT_EXTENT = [113.0, 124.0, 29.0, 40.0]  # lon_min, lon_max, lat_min, lat_max


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one raw Huadong radar PNG map")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input PNG file.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--province_shp", default=DEFAULT_PROVINCE_SHP)
    parser.add_argument("--city_shp", default=DEFAULT_CITY_SHP)
    parser.add_argument("--background", default=DEFAULT_BACKGROUND)
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        default=DEFAULT_EXTENT,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input PNG not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_huadong_observation.png"

    projection, provinces, cities, background_image = load_map_resources(
        Path(args.province_shp), Path(args.city_shp), Path(args.background)
    )
    save_map(
        read_dbz(input_path),
        output_path,
        projection,
        provinces,
        cities,
        background_image,
        args.extent,
    )
    print(f"Saved Huadong observation map: {output_path.resolve()}")


if __name__ == "__main__":
    main()
