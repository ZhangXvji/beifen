#!/usr/bin/env python

#我不需要一个可视化的BEV图，而是一个以x、y坐标作为索引，高度、反射率和点密度作为对应索引的值的高维矩阵
#即一个类似BEV的数据结构，每个像素位置包含高度、反射率、点密度等信息
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import numpy as np
from collections import defaultdict
import open3d as o3d

# 鸟瞰图参数
bev_resolution = 0.1  # 鸟瞰图像素与实际距离的比例
bev_size = (800, 800)  # 鸟瞰图像素尺寸

def transform_to_bev_coordinate(point, vehicle_pose):
    # 假设vehicle_pose包含车辆的位置和朝向信息
    # 进行坐标变换，将点云从车体坐标系转换到鸟瞰视图坐标系
    # 在实际应用中，您需要使用车辆的位置和朝向来进行转换
    bev_x = int((point[1] - vehicle_pose[0]) / bev_resolution) + bev_size[0] // 2
    bev_y = int((point[0] - vehicle_pose[1]) / bev_resolution) + bev_size[1] // 2
    return bev_x, bev_y

def process_pointcloud(data):
    assert isinstance(data, PointCloud2)
    gen = point_cloud2.read_points(data)
    
    # 创建高纬矩阵，每个像素位置存储高度、反射率和点密度信息
    bev_matrix = defaultdict(lambda: {"height": [], "intensity": [], "point_count": 0})
    
    for p in gen:
    #形如(4.777646064758301, -0.010840152390301228, 0.5819261074066162, 65.0, 11, 0.10004305839538574)=(x,y,z,i,r,t)
        x, y, z, intensity, _, _ = p

        vehicle_pose = (0, 0)
        bev_x, bev_y = transform_to_bev_coordinate((x, y, z), vehicle_pose)

        if 0 <= bev_x < bev_size[0] and 0 <= bev_y < bev_size[1]:
            pixel_info = bev_matrix[(bev_x, bev_y)]
            pixel_info["height"].append(z)
            pixel_info["intensity"].append(intensity)
            pixel_info["point_count"] += 1
    
    # 将bev_matrix转换为三个矩阵，分别存储高度、反射率和点密度信息
    height_matrix = np.zeros(bev_size)
    intensity_matrix = np.zeros(bev_size)
    density_matrix = np.zeros(bev_size)

    for pixel, info in bev_matrix.items():
        if info["point_count"] > 0:
            avg_height = sum(info["height"]) / info["point_count"]
            avg_intensity = sum(info["intensity"]) / info["point_count"]

            height_matrix[pixel[1], pixel[0]] = avg_height
            intensity_matrix[pixel[1], pixel[0]] = avg_intensity
            density_matrix[pixel[1], pixel[0]] = info["point_count"]

    return height_matrix, intensity_matrix, density_matrix

# ...

# 在回调函数callback_pointcloud中使用process_pointcloud函数
def callback_pointcloud(data):
    height_matrix, intensity_matrix, density_matrix = process_pointcloud(data)

    # 将高度、反射率和点密度矩阵拼接起来
    # 拼成一个类似图片的结构，shape = (800,800,3)
    stacked_matrix = np.stack((height_matrix, intensity_matrix, density_matrix), axis=-1)
    return(stacked_matrix)


