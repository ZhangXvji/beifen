#!/usr/bin/env python

import os

directory_path = "../dataset/raw/"
folder_dict = {} #这个字典的键是一组数据存储文件夹的名称(也是rosbag的包名)，值是总帧数
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
folder_dict = {key: value // 10 -7 for key, value in folder_dict.items()}

print(folder_dict)

# 获取第一个键和对应的值
first_key = next(iter(folder_dict))

first_value = folder_dict[first_key]

# 计算所有值的和
total_value = sum(folder_dict.values())



# 定义索引 index
index = 3000  # 你可以将 index 替换为你需要的值

# 初始化 bag_time 为空字符串
bag_time = ""

# 遍历字典 self_dict，计算 bag_time
current_sum = 0
for key, value in folder_dict.items():
    current_sum += value
    # if index < current_sum:
    bag_time = key


        # break