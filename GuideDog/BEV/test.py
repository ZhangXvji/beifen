#!/usr/bin/env python

#bev语义信息实际含义
#在COCO Classes 的基础上+1,如 1: person
#0: 未检测

# COCO Classes
#   0: person
#   26: handbag
#   56: chair
#   60: dining table
#   62: tv
#   63: laptop
#   64: mouse

# 鸟瞰图参数
bev_resolution = 0.1  # 鸟瞰图像素与实际距离的比例
bev_size = (200, 200)  # 鸟瞰图像素尺寸 
# 实际约 20m x 20m

import sys
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from collections import defaultdict
import os
import time
import threading

# 定义一个全局变量，用于存储结果
result_lock = threading.Lock()
result_matrix = None
# 指定标定参数： 内参/外参
#左左(能看到腿)
intrinsic_matrix0 = np.array([[478.837,0,321.968],[0,478.908,233.622],[0,0,1]])
extrinsic_matrix0 = np.array([[0.998332,-0.0440726,-0.0373049,-0.0256169],[-0.0345651,0.0613706,-0.997516,0.0229135],[0.0462526,0.997142,0.0597448,-0.0879064],[0,0,0,1]])
#左
intrinsic_matrix2 = np.array([[491.555,0,306.777],[0,491.285,236.629],[0,0,1]])
extrinsic_matrix2 = np.array([[0.48775,-0.872763,-0.019611,-0.0185514],[-0.0483464,-0.00457523,-0.99882,0.0688649],[0.871644,0.488123,-0.0444265,-0.0303459],[0,0,0,1]])
#右
intrinsic_matrix4 = np.array([[487.703,0,302.612],[0,488.584,242.112],[0,0,1]])
extrinsic_matrix4 = np.array([[-0.510725,-0.85871,-0.0421568,-0.0140267],[0.0458616,0.0217534,-0.998711,0.0505157],[0.85852,-0.512,0.0282718,-0.0485396],[0,0,0,1]])
#右右
intrinsic_matrix6 = np.array([[486.296,0,332.114],[0,487.392,259.78],[0,0,1]])
extrinsic_matrix6 = np.array([[-0.999508,-0.0311525,-0.00368919,-0.0997821],[0.00107343,0.0835688,-0.996501,0.180676],[0.0313518,-0.996015,-0.0834942,-0.0784604],[0,0,0,1]])

def callback_image(path_img):
    cv_image = np.array(cv2.imread(path_img))

    # 将OpenCV格式图片转Numpy数组
    np_image = np.array(cv_image)
     
    height, width, _ = np_image.shape
    result_image = np.zeros((height, width, 3), dtype=np.uint8)
     
    model = YOLO('yolov8n-seg.pt')

    results = model(np_image)
    mask_result = np.zeros((height, width, 1), dtype=np.uint8)

    for r in results:
        if(len(r.boxes.cls) != 0):
            for i in range(len(r.boxes.cls)):
                new_mask = np.zeros((height, width, 1), dtype=np.uint8)
                new_mask[r.masks.data[i].cpu().numpy().reshape((480,640,1))[..., 0] != 0] = r.boxes.cls[i].cpu().numpy()+1
                # print(r.masks.data[i].cpu().numpy().shape)
                mask_result = mask_result + new_mask
    return(mask_result)

def transform_to_bev_coordinate(point, vehicle_pose):
    # 假设vehicle_pose包含车辆的位置和朝向信息
    # 进行坐标变换，将点云从车体坐标系转换到鸟瞰视图坐标系
    # 在实际应用中，您需要使用车辆的位置和朝向来进行转换
    bev_x = int((point[1] - vehicle_pose[0]) / bev_resolution) + bev_size[0] // 2
    bev_y = int((point[0] - vehicle_pose[1]) / bev_resolution) + bev_size[1] // 2
    return bev_x, bev_y

def thread_function(camera_id, intrinsic_matrix, extrinsic_matrix, img_msg, point_cloud):
    global result_matrix

    # 调用 get_semantic_mag 函数计算语义矩阵
    semantic_matrix = get_semantic_mag(intrinsic_matrix, extrinsic_matrix, img_msg, point_cloud, camera_id)

    # 使用锁来确保多个线程安全地更新结果
    with result_lock:
        if result_matrix is None:
            result_matrix = semantic_matrix
        else:
            # 合并语义矩阵，这里使用 np.maximum 函数
            result_matrix = np.maximum(result_matrix, semantic_matrix)


def get_semantic_mag(intrinsic_matrix,extrinsic_matrix,img_msg, point_cloud, camera_id):
    point_cloud = o3d.io.read_point_cloud(point_cloud)
    # 将激光雷达点云从open3d格式转换为numpy数组
    point_cloud_numpy = np.asarray(point_cloud.points)
    # 添加齐次坐标
    homogeneous_points = np.hstack((point_cloud_numpy, np.ones((point_cloud_numpy.shape[0], 1))))
    # 计算投影坐标
    projected_points = np.dot(homogeneous_points, extrinsic_matrix.T)[:, :3]
    projected_points = np.dot(projected_points, intrinsic_matrix.T)
    # 透视除法
    # projected_points /= projected_points[:, 2][:, np.newaxis]
    projected_points[:, 0] /= projected_points[:, 2]
    projected_points[:, 1] /= projected_points[:, 2]
    # 获取语义信息并设置到点云中
    semantics = []
    semantic_mag = callback_image(f"../dataset/raw/2023-08-19-20-14-45/video/{camera_id}/1.jpg")
    # print(semantic_mag)
    for i in range(projected_points.shape[0]):
        if projected_points[i,2]>0:
            u = int(projected_points[i, 0])
            v = int(projected_points[i, 1])
            if 0 <= u < 56 and 0 <= v < 56:
                semantic = int(semantic_mag[v, u])
                # print(semantic)
                semantics.append(semantic)
            else:
                semantics.append(0)  # 默认黑色
        else:
            semantics.append(0)  # 默认黑色
    semantics_array = np.array(semantics).reshape(-1, 1)
    points_with_semantic = np.hstack((np.array(point_cloud.points), semantics_array))
    
    # 创建语义BEV
    h, w = bev_size
    semantic_matrix = np.zeros((h, w, 1), dtype=np.uint8)

    for p in points_with_semantic:
        x, y, z, sem = p
        if(sem != 0):
            # print(f"x:{x},y:{y},z:{z},sem:{sem}")
            vehicle_pose = (0,0) # 以激光雷达坐标系建系，不整imu是因为imu-lidar的外参标定精度难以保证
            bev_x, bev_y = transform_to_bev_coordinate((x,y,z), vehicle_pose)
            if 0 <= bev_x < bev_size[0] and 0 <= bev_y < bev_size[1]:
                semantic_matrix[int(bev_y), int(bev_x)] = sem

    # print(semantic_matrix.shape)
    # plt.imshow(semantic_matrix.squeeze(), cmap='gray')  # 使用squeeze()将单通道维度移除
    # plt.axis('off')
    # plt.show()
    return(semantic_matrix)

    ############3可视化测试###############
    # semantics = []
    # semantic_mag = callback_image(f"../dataset/raw/{bag_time}/video/{camera_id}/{time}.jpg")
    # for i in range(projected_points.shape[0]):
    #     if projected_points[i,2]>0:
    #         u = int(projected_points[i, 0])
    #         v = int(projected_points[i, 1])
    #         if 0 <= u < 640 and 0 <= v < 480:
    #             senmatic = semantic_mag[v, u]
    #             semantics.append([0,0,0])
    #         else:
    #             semantics.append([255,255,255])  # 默认黑色
    #     else:
    #         semantics.append([255,255,255])  # 默认黑色
    # point_cloud.colors = o3d.utility.Vector3dVector(semantics)
    # o3d.visualization.draw_geometries([point_cloud]) # 可视化

def main(img1_msg, img2_msg, img3_msg, img4_msg, point_cloud):
    # 初始化所有数据

    # 创建四个线程，每个线程负责一个语义矩阵的计算
    thread0 = threading.Thread(target=thread_function, args=(0, intrinsic_matrix6, extrinsic_matrix6, img1_msg, point_cloud))
    thread2 = threading.Thread(target=thread_function, args=(2, intrinsic_matrix4, extrinsic_matrix4, img2_msg, point_cloud))
    thread4 = threading.Thread(target=thread_function, args=(4, intrinsic_matrix2, extrinsic_matrix2, img3_msg, point_cloud))
    thread6 = threading.Thread(target=thread_function, args=(6, intrinsic_matrix0, extrinsic_matrix0, img4_msg, point_cloud))

    # 启动线程
    thread0.start()
    thread2.start()
    thread4.start()
    thread6.start()

    # 等待所有线程结束
    thread0.join()
    thread2.join()
    thread4.join()
    thread6.join()

    # 获取合并后的语义矩阵
    global result_matrix
    semantic_matrix = result_matrix

    return semantic_matrix

if __name__ == "__main__":

    start_time = time.time()
    main("../dataset/raw/2023-08-19-20-14-45/video/0/1.jpg",
         "../dataset/raw/2023-08-19-20-14-45/video/2/1.jpg",
         "../dataset/raw/2023-08-19-20-14-45/video/4/1.jpg",
         "../dataset/raw/2023-08-19-20-14-45/video/6/1.jpg",
         "../dataset/raw/2023-08-19-20-14-45/cloudpoints/1.pcd")
    end_time = time.time()

    # 计算程序执行时间
    execution_time = end_time - start_time

    # 打印执行时间（以秒为单位）
    print("程序执行时间：", execution_time, "秒")