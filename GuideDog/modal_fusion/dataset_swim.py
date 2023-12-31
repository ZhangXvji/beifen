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
# Ignore warnings
import warnings

import cv2

warnings.filterwarnings("ignore")


# 定义数据集类
class DogDataset(Dataset):
    def __init__(self):
        
        directory_path = "../dataset/raw/"
        folder_dict = {}
        folders = os.listdir(directory_path)
        for folder in folders:
            folder_path = os.path.join(directory_path, folder)
            if os.path.isdir(folder_path):
                folder_lidar = os.listdir(folder_path+"/cloudpoints/")
                frame_num = len(folder_lidar)
                folder_dict[folder] = frame_num
        # 删除所有值小于70的键
        folder_dict = {key: value for key, value in folder_dict.items() if value >= 70}
        # 将剩余键的值整除10
        folder_dict = {key: value // 10 -6  for key, value in folder_dict.items()}

        self.dict = folder_dict # 这个字典的键是一组数据存储文件夹的名称(也是rosbag的包名)，值是总帧数
        # 形如{'2023-08-23-15-20-07': 5317, '2023-08-23-12-33-26': 11619}


    def __len__(self):
        return sum(self.dict.values())


    def __getitem__(self, index):
        bag_time, index_in_bag = self.get_bag_time(index)
        train_range = [int(10*index_in_bag-9), int(10*index_in_bag+50)]
        label_range = [int(10*index_in_bag+51), int(10*index_in_bag+60)]

        rgb0 = self.get_video_tensor(bag_time, train_range, 0)
        rgb2 = self.get_video_tensor(bag_time, train_range, 2)
        rgb4 = self.get_video_tensor(bag_time, train_range, 4)
        rgb6 = self.get_video_tensor(bag_time, train_range, 6)
        video = self.get_video_tensor(bag_time, train_range, "leg") # 3,60,56,56

        label = self.get_label_tensor(bag_time, label_range) # 20

        sample = {'rgb0' : rgb0.float(),
                  'rgb2' : rgb2.float(),
                  'rgb4' : rgb4.float(),
                  'rgb6' : rgb6.float(),
                  'video': video.float(),
                  'label': label.float()}
        return sample
    
    def get_bag_time(self, index):
        # 遍历字典 self_dict，计算 bag_time
        current_sum = 0
        for key, value in self.dict.items():
            current_sum += value
            if index < current_sum:
                bag_time = key
                index_in_bag = index+1 - (current_sum - value)
                return bag_time, index_in_bag
                break


    def get_video_tensor(self, bag_time, train_range,code):
        image_folder = f"../dataset/raw/{bag_time}/video/{code}/"
        target_height = 56
        target_width = 56
        num_channels = 3
        image_tensors = []
        # 读取图像并将其转换为PyTorch张量
        for i in range(train_range[0], train_range[1] + 1):
            image_path = os.path.join(image_folder, f"{i}.jpg")
            image = Image.open(image_path).convert("RGB") 
            image = image.resize((target_width, target_height))  # 调整图像尺寸
            image_array = np.array(image, dtype=np.float32) / 255.0
            image_tensor = torch.tensor(image_array, dtype=torch.float32)
            image_tensors.append(image_tensor)


        # 将图像张量堆叠成一个批次
        image_batch =  torch.stack(image_tensors) #60x56X56X3 DHWC
        image_batch = torch.permute(image_batch, (3,0,1,2)) # 3,60,56,56
        return image_batch
        # # 打印结果的形状
        # print(image_batch)
        # print(image_batch.shape)

    def get_label_tensor(self, bag_time, label_range):
        path = f"../dataset/raw/{bag_time}/ecparm/motor/motor_raw.csv"
        data = pd.read_csv(path)
        # 提取10行的数据
        data_subset = data.iloc[label_range[0]-1:label_range[1], 1:]
        # 将数据转换为NumPy数组
        data_array = data_subset.to_numpy()

        data_tensor = []
        for i in range(10):
            count = self.define_classed(data_array[i][0],data_array[i][1])
            if count == 0:
                label = [1,0,0,0,0]
            elif count == 1:
                label = [0,1,0,0,0]
            elif count == 2:
                label = [0,0,1,0,0]
            elif count == 3:
                label = [0,0,0,1,0]
            else:
                label = [0,0,0,0,1]
            data_tensor.append(label)
        data_tensor = torch.tensor(data_tensor)
            
        
        # 将NumPy数组转换为PyTorch张量
        # tensor_data = torch.tensor(data_array, dtype=torch.float32).reshape(20)
        return(data_tensor)
        # return tensor_data
        # # 打印张量的形状
        # print(tensor_data)
        # print(tensor_data.shape) #20

    def define_classed(self, m1, m2):
        if np.logical_or(abs(m1) < 250, abs(m2) < 250):
            return 0 # 停止
        elif m1 >= 250 and m2 >= 250 and abs(m1 - m2) < 250:
            return 1 # 前进
        elif m1 <= -250 and m2 <= 250 and abs(m1 - m2) < 250:
            return 2 # 后退
        elif m1 - m2 >= 250:
            return 3 # 右转
        else:
            return 4 # 左转

    def custom_sort_key(filename):
        return float(filename[:-4])  # 去掉文件扩展名后转换为浮点数
    
if __name__ == "__main__":
    test_dataset = DogDataset()
    # test_dataset.get_imu_tensor("2023-08-19-20-14-45",[1,60])
    # test_dataset.get_video_tensor("2023-08-19-20-14-45",[1,60])
    # # test_dataset.get_sensor_tensor("2023-08-19-20-14-45",[1,60])
    # test_dataset.get_motor_tensor("2023-08-19-20-14-45",[1,60])
    # test_dataset.get_label_tensor("2023-08-19-20-14-45",[61,70])
    # test_dataset.get_bev_tensor("2023-08-19-20-14-45",[1,60]) 
    # a = test_dataset.get_video_tensor("2023-08-24-18-20-06",[1,60], 0)  
    # print(a)