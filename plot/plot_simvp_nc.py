from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.colors import BoundaryNorm, ListedColormap
from netCDF4 import Dataset, num2date


DEFAULT_INPUT = "inference/mix/simvp_202508241000.nc"
DEFAULT_PROVINCE_SHP = r"D:\Code\Data\map\bou2_4l.shp"
DEFAULT_CITY_SHP = r"D:\Code\Data\map\xinjiang_city\xinjiang_city.shp"
DEFAULT_BACKGROUND = r"D:\Code\Data\map\NE1_50M_SR_W.tif"
DEFAULT_EXTENT = [78.0, 89.0, 39.0, 50.0]  # lon_min, lon_max, lat_min, lat_max

LEVELS = [-32768, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
COLORS = [
    "#FFFFFF",
    "#1E90FF",
    "#01F508",
    "#00A433",
    "#008000",
    "#FFFF00",
    "#FFDC01",
    "#FFA500",
    "#FF0000",
    "#B22222",
    "#8B0000",
    "#FF00FF",
    "#8B008B",
]
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm(LEVELS, CMAP.N, clip=True)


def parse_args(default_extent=DEFAULT_EXTENT) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot all SimVP NetCDF forecast maps")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input NetCDF file.")
    parser.add_argument("--frames_dir", default=None, help="Directory for all forecast maps.")
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


def read_forecast(path: Path):
    with Dataset(path) as nc:
        dbz = np.asarray(nc.variables["forecast_dbz"][:], dtype=np.float32)
        time_var = nc.variables["time"]
        times = num2date(
            time_var[:],
            units=time_var.units,
            calendar=getattr(time_var, "calendar", "standard"),
            only_use_cftime_datetimes=False,
        )
        if "lead_time" in nc.variables:
            leads = np.asarray(nc.variables["lead_time"][:], dtype=np.int32)
        else:
            leads = np.arange(1, dbz.shape[0] + 1, dtype=np.int32) * 6
        reference_time = getattr(nc, "forecast_reference_time", "")
    return dbz, list(times), leads, reference_time


def load_map_resources(province_shp: Path, city_shp: Path, background: Path):
    for path in (province_shp, city_shp, background):
        if not path.is_file():
            raise FileNotFoundError(f"Map dependency not found: {path}")

    projection = ccrs.PlateCarree()
    provinces = cfeature.ShapelyFeature(
        list(Reader(str(province_shp)).geometries()),
        projection,
        edgecolor="black",
        facecolor="none",
    )
    cities = cfeature.ShapelyFeature(
        list(Reader(str(city_shp)).geometries()),
        projection,
        edgecolor="black",
        facecolor="none",
    )
    background_image = plt.imread(background)
    return projection, provinces, cities, background_image


def setup_map(ax, projection, provinces, cities, background_image, extent, labels: bool):
    lon_min, lon_max, lat_min, lat_max = extent
    ax.set_extent(extent, crs=projection)
    ax.imshow(
        background_image,
        origin="upper",
        transform=projection,
        extent=[-180, 180, -90, 90],
        zorder=0,
    )
    ax.add_feature(provinces, linewidth=0.9, zorder=3)
    ax.add_feature(cities, linewidth=0.5, linestyle="--", zorder=3)

    x_ticks = np.arange(np.ceil(lon_min / 2) * 2, lon_max + 0.1, 2)
    y_ticks = np.arange(np.ceil(lat_min / 2) * 2, lat_max + 0.1, 2)
    ax.gridlines(
        crs=projection,
        draw_labels=False,
        linewidth=0.5,
        color="black",
        alpha=0.35,
        linestyle="--",
        xlocs=x_ticks,
        ylocs=y_ticks,
        zorder=4,
    )
    if labels:
        ax.set_xticks(x_ticks, crs=projection)
        ax.set_yticks(y_ticks, crs=projection)
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(labelsize=20)


def draw_radar(ax, frame, projection, extent):
    return ax.imshow(
        frame,
        cmap=CMAP,
        norm=NORM,
        origin="lower",
        extent=extent,
        transform=projection,
        interpolation="nearest",
        zorder=2,
    )


def save_all_forecast_maps(
    dbz,
    times,
    leads,
    output_dir,
    projection,
    provinces,
    cities,
    background_image,
    extent,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(dbz.shape[0]):
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        setup_map(ax, projection, provinces, cities, background_image, extent, labels=True)
        image = draw_radar(ax, dbz[index], projection, extent)
        colorbar = fig.colorbar(image, ax=ax, ticks=LEVELS[1:], shrink=0.82, pad=0.04)
        colorbar.set_label("Reflectivity (dBZ)")
        filename = f"forecast_{times[index]:%Y%m%d%H%M}_f{int(leads[index]):03d}.png"
        fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)


def main(default_extent=DEFAULT_EXTENT, output_tag="forecast_maps") -> None:
    args = parse_args(default_extent)
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"NetCDF file not found: {input_path}")

    frames_dir = Path(args.frames_dir) if args.frames_dir else Path("plot/output") / (
        input_path.stem + f"_{output_tag}"
    )

    dbz, times, leads, _ = read_forecast(input_path)
    projection, provinces, cities, background_image = load_map_resources(
        Path(args.province_shp), Path(args.city_shp), Path(args.background)
    )
    save_all_forecast_maps(
        dbz,
        times,
        leads,
        frames_dir,
        projection,
        provinces,
        cities,
        background_image,
        args.extent,
    )
    print(f"Saved {dbz.shape[0]} forecast maps: {frames_dir.resolve()}")


if __name__ == "__main__":
    main()
