import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import BoundaryNorm,ListedColormap
from matplotlib.image import imread
import matplotlib
import pandas as pd
import os
import datetime as dt
from fanjiang.utils import add_time

import multiprocessing as mp

matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.weight'] = "bold"
prov_shp = shpreader.Reader(r"D:\Code\Data\map\bou2_4l.shp")
city_shp = shpreader.Reader("D:\Code\Data\map\city.shp")
levels = [0, 0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
cmap_dbz = ListedColormap(['#FFFFFF', "dodgerblue",
                           "#01F508", "#00A433", "green", "yellow", "#FFDC01",
                           "orange", "red", "firebrick",
                           "darkred", "magenta", "darkmagenta"])
norm = BoundaryNorm(levels, ncolors=cmap_dbz.N, clip=True)

lat_min = 30.4
lat_max = 35.2
lon_min = 116.33
lon_max = 121.93
region = lat_max, lat_min, lon_min, lon_max


def get_time(radar_f, postfix="_qc.nc"):
    name = os.path.basename(radar_f)
    name = name.rstrip(postfix)
    time = name[-12:]
    return time


def plot(radar_f):

    def plot_single(img, region, title, save_f):
        img_h, img_w = img.shape[:2]

        lat_max, lat_min, lon_min, lon_max = region
        lat1d = np.linspace(lat_min + 0.01, lat_max, img_h)
        lon1d = np.linspace(lon_min + 0.01, lon_max, img_w)
        grid_lat, grid_lon = np.meshgrid(lat1d, lon1d, indexing="ij")

        projection = ccrs.PlateCarree()
        fig = plt.figure(figsize=[15, 12])
        ax1 = fig.add_subplot(1, 1, 1, projection=projection)
        ax1.add_feature(cfeature.LAKES.with_scale('50m'), zorder=4, alpha=0.3)
        ax1.add_feature(cfeature.RIVERS.with_scale('50m'), zorder=5, alpha=0.5)

        ax1.add_feature(cfeature.ShapelyFeature(prov_shp.geometries(), projection, \
                                                    edgecolor='k', facecolor='none'),
                            linewidth=2.5, linestyle='-', zorder=4, alpha=0.8)

        ax1.add_feature(cfeature.ShapelyFeature(city_shp.geometries(), projection, \
                                                    edgecolor='k', facecolor='none'),
                            linewidth=1., linestyle='-', zorder=4, alpha=0.8)

        ax1.set_extent([lon_min, lon_max, lat_min, lat_max], projection)


        ax1.gridlines(crs=ccrs.PlateCarree(),
            draw_labels=False,
            linewidth=1.2,
            color='k',
            alpha=0.5,
            linestyle='--',
            zorder=8)


        gci1 = ax1.pcolormesh(grid_lon, grid_lat, img, cmap=cmap_dbz, norm=norm, zorder=3, alpha=1)
        ax1.set_xticks(np.arange(np.round(lon_min,0), np.round(lon_max,0), 2), crs=projection)
        ax1.set_xticklabels(np.arange(np.round(lon_min,0), np.round(lon_max,0), 2), fontsize=22)
        ax1.set_yticks(np.arange(np.round(lat_min+0.5,0), np.round(lat_max+0.5,0), 1), crs=projection)
        ax1.set_yticklabels(np.arange(np.round(lat_min+0.5,0), np.round(lat_max+0.5,0), 1), fontsize=22)

        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax1.xaxis.set_major_formatter(lon_formatter)
        ax1.yaxis.set_major_formatter(lat_formatter)
        cb = plt.colorbar(gci1, shrink = 0.9)

        cb.set_ticks(levels)
        cb.ax.set_yticklabels(levels, fontsize=22)
        cb.set_label("Reflectivity (dBZ)", fontsize=28)

        plt.title(title, fontsize=25)
        plt.savefig(save_f, bbox_inches='tight', pad_inches=0.0, dpi=200)
        plt.close()

    img = cv2.imread(radar_f, 0)
    # img = img/255*70  # 真值
    plot_single(img, region, title=os.path.basename(radar_f), save_f=radar_f.replace('.png', '_new.png'))

def main(model):
    if model == 'single':
        radar_f = r"D:\Code\Tianchi\data\TestB1_infer\Radar\001\radar_00_.png"
        plot(radar_f)

    elif model == 'multi':
        pool = mp.Pool(4)
        files_dir = "D:\Code\Tianchi\data\TestB1_infer\Precip"
        files = glob.glob(os.path.join(files_dir, '*', '*'))
        for file in files:
            print(file)
            pool.apply_async(plot, args=(file,))

        pool.close()
        pool.join()  # 等待子进程结束


if __name__ == '__main__':
    model = 'multi'
    main(model)





