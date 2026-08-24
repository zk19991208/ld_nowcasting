import pandas as pd
from datetime import datetime
import numpy as np
import os

file = "/home/zhengyu/preprocess/EastChina1km/data/datetime.txt"

radar_dir = "/home/zhengyu/dataset/EastChinaPR480x560/CR"
prec_dir = "/home/zhengyu/dataset/EastChinaPR480x560/Prec"

parsers = lambda x: datetime.strptime(x, "%Y/%m/%d %H:%M:%S")
df1 = pd.read_csv(file, header=None, names=["time",], parse_dates=["time"], date_parser=parsers) #UTC
print(df1.shape)
df = df1.drop_duplicates(["time"]).sort_values(by="time").reset_index()
csv_sample = []
len_samples = df["time"].shape[0]
istart = 0
time_series = []
count = 0

while istart < len_samples:
    start_time = df["time"][istart]
    for j in range(istart, len_samples):
        if not (start_time + pd.Timedelta(j-istart, "H")) == df["time"][j]:
            istart = j
            break

        if j == (len_samples - 1):
            istart = j + 1
            break
    end_time = df["time"][istart-1]
    series = pd.date_range(start_time, end_time, freq="6min")
    if len(series) >= 20:
        count+=1
        time_series.append(series)
        for i in range(0, len(series)-19, 1):
            csv_sample.append(pd.date_range(series[i] - pd.Timedelta(1,"h"), periods=50, freq="6min").strftime("%Y%m%d%H%M"))#series[i:i+50].strftime("%Y%m%d%H%M"))

dataset = pd.DataFrame(csv_sample)
nrows, ncols = dataset.shape
index_train = np.zeros(nrows, dtype=bool)

for icol in range(ncols):
    dataset.iloc[:, icol] = pd.to_datetime(dataset.loc[:, icol]).dt.strftime("%Y/%m/%Y%m%d%H%M.png")

for i in range(nrows):
    count = 0
    for j in range(ncols):
        file_radar = os.path.join(radar_dir, dataset.iloc[i, j])
        file_prec = os.path.join(prec_dir, dataset.iloc[i, j])
        if not os.path.exists(file_radar):
            break
        if not os.path.exists(file_prec):
            break
        count += 1
    print(count)
    if count == 50:
        index_train[i] = True

dataset = dataset[index_train]
dataset_train = dataset.loc[:75000]
dataset_val = dataset.loc[75000:80600]
dataset_test = dataset.loc[80600:]


dataset.to_csv("/home/zhuangxr/work/zengkang/data/dataset_new_50.csv", header=None, index=None)
dataset_train.to_csv("/home/zhuangxr/work/zengkang/data/dataset_train_new_50.csv", header=None, index=None)
dataset_val.to_csv("/home/zhuangxr/work/zengkang/data/dataset_val_new_50.csv", header=None, index=None)
dataset_test.to_csv("/home/zhuangxr/work/zengkang/data/dataset_test_new_50.csv", header=None, index=None)

times_rain = pd.DataFrame(np.concatenate(time_series))
times_rain.to_csv("/home/zhuangxr/work/zengkang/data/dataset_series_new_50.csv", header=None, index=None)
