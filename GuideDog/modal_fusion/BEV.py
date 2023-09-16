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
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image, PointCloud2, CompressedImage, Imu
from message_filters import ApproximateTimeSynchronizer, Subscriber
import ecparm
import pcl
import threading

def callback_image(np_image):
    # 将OpenCV格式图片转Numpy数组
    height, width, _ = np_image.shape
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

def semantic_matrix(img1_msg, img2_msg, img3_msg, img4_msg, point_cloud):
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

    semantic_matrix_0 = get_semantic_mag(intrinsic_matrix0,extrinsic_matrix0,img1_msg, point_cloud)
    semantic_matrix_2 = get_semantic_mag(intrinsic_matrix2,extrinsic_matrix2,img2_msg, point_cloud)
    semantic_matrix_4 = get_semantic_mag(intrinsic_matrix4,extrinsic_matrix4,img3_msg, point_cloud)
    semantic_matrix_6 = get_semantic_mag(intrinsic_matrix6,extrinsic_matrix6,img4_msg, point_cloud)
    #语义BEV
    semantic_matrix = np.maximum.reduce([semantic_matrix_0, semantic_matrix_2, semantic_matrix_4, semantic_matrix_6])
    return(semantic_matrix)


def get_semantic_mag(intrinsic_matrix,extrinsic_matrix,img_msg, point_cloud_numpy):
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
    semantic_mag = callback_image(img_msg)
    # print(semantic_mag)
    for i in range(projected_points.shape[0]):
        if projected_points[i,2]>0:
            u = int(projected_points[i, 0])
            v = int(projected_points[i, 1])
            if 0 <= u < 640 and 0 <= v < 480:
                semantic = int(semantic_mag[v, u])
                # print(semantic)
                semantics.append(semantic)
            else:
                semantics.append(0)  # 默认黑色
        else:
            semantics.append(0)  # 默认黑色
    semantics_array = np.array(semantics).reshape(-1, 1)
    points_with_semantic = np.hstack((point_cloud_numpy, semantics_array))
    
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

    return(semantic_matrix)

