#!/usr/bin/env python
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import io, transforms
import pandas as pd
import numpy as np
import pandas as pd
pd.set_option('display.width',None)#设置数据展示宽度
# Ignore warnings
import warnings
import csv
import cv2
import ast
from sklearn import preprocessing
import h5py

warnings.filterwarnings("ignore")


# 定义数据集类
class DogDataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        self.group_names = None  # 将数据集的初始化延迟到__getitem__中

    def __len__(self):
        if self.group_names is None:
            # 打开HDF5文件并获取组名列表
            with h5py.File(self.hdf5_file, 'r') as file:
                self.group_names = list(file.keys())
        return len(self.group_names)


    def __getitem__(self, index):
        if self.group_names is None:
            # 打开HDF5文件并获取组名列表
            with h5py.File(self.hdf5_file, 'r') as file:
                self.group_names = list(file.keys())

        group_name = self.group_names[index]
        with h5py.File(self.hdf5_file, 'r') as file:
            group = file[group_name]

            video0 = torch.tensor(group['video0'][()])
            video2 = torch.tensor(group['video2'][()])
            video4 = torch.tensor(group['video4'][()])
            video6 = torch.tensor(group['video6'][()])
            videoL = torch.tensor(group['videoL'][()])
            imu = torch.tensor(group['imu'][()])
            motor = torch.tensor(group['motor'][()])
            label = torch.tensor(group['label'][()])
            class_value = torch.tensor(group['class'][()])

        sample = {
            'video0': video0,
            'video2': video2,
            'video4': video4,
            'video6': video6,
            'videoL': videoL,
            'imu': imu,
            'motor': motor,
            'label': label,
            'class': class_value
        }

        return sample

if __name__ == "__main__":
    hdf5_file = "data.hdf5"  # 将文件名替换为你的HDF5文件路径
    dataset = DogDataset(hdf5_file)

    len = dataset.__len__()

    class_0 = 0
    class_1 = 0
    class_2 = 0
    class_3 = 0
    class_4 = 0

    for index in range(len):
        sample = dataset[index]
        if sample['class'] == 0:
            class_0 = class_0 +1
        elif sample['class'] == 1:
            class_1 = class_1 +1
        elif sample['class'] == 2:
            class_2 = class_2 +1
        elif sample['class'] == 3:
            class_3 = class_3 +1
        else:
            class_4 = class_4 +1

    print(f"停止 ： {class_0}")
    print(f"前进 ： {class_1}")
    print(f"后退 ： {class_2}")
    print(f"右转 ： {class_3}")
    print(f"左转 ： {class_4}")