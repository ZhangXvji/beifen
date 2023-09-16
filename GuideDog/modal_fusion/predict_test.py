#!/usr/bin/env python

import BEV
import torch
import multi_swin
from dataset import DogDataset
from torch.utils.data import DataLoader
import rospy
from sensor_msgs.msg import Image, PointCloud2, CompressedImage, Imu
from sensor_msgs import point_cloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from ecparm.msg import ECParm
from ecparm.msg import Motor
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO
import threading
from cv_bridge import CvBridge
from queue import Queue


hz = 10
max_length = 60

image_leg_queue = deque(maxlen=max_length)
imu_queue = deque(maxlen=max_length)
motor_queue = deque(maxlen=max_length)

img1_queue = deque(maxlen=max_length)
img2_queue = deque(maxlen=max_length)
img3_queue = deque(maxlen=max_length)
img4_queue = deque(maxlen=max_length)
point_cloud_queue = deque(maxlen=max_length)


def image1_callback(msg):
    rate = rospy.Rate(hz)
    # 使用 cv_bridge 将压缩图像消息解码为图像
    bridge = CvBridge()
    cv_image = bridge.compressed_imgmsg_to_cv2(msg, "rgb8")
    np_image = np.array(cv_image)
    np_image = np_image.transpose(2, 0, 1)
    img1_queue.append(np_image)
    print(f"image1 : {len(img1_queue)}")
    rate.sleep()

def image2_callback(msg):
    rate = rospy.Rate(hz)
    bridge = CvBridge()
    cv_image = bridge.compressed_imgmsg_to_cv2(msg, "rgb8")
    np_image = np.array(cv_image)
    np_image = np_image.transpose(2, 0, 1)
    img2_queue.append(np_image)
    print(f"image2 : {len(img2_queue)}")
    rate.sleep()

def image3_callback(msg):
    rate = rospy.Rate(hz)
    bridge = CvBridge()
    cv_image = bridge.compressed_imgmsg_to_cv2(msg, "rgb8")
    np_image = np.array(cv_image)
    np_image = np_image.transpose(2, 0, 1)
    img3_queue.append(np_image)
    print(f"image3 : {len(img3_queue)}")
    rate.sleep()

def image4_callback(msg):
    rate = rospy.Rate(hz)
    bridge = CvBridge()
    cv_image = bridge.compressed_imgmsg_to_cv2(msg, "rgb8")
    np_image = np.array(cv_image)
    np_image = np_image.transpose(2, 0, 1)
    img4_queue.append(np_image)
    print(f"image4 : {len(img4_queue)}")
    rate.sleep()

# def pointcloud_callback(msg):
#     rate = rospy.Rate(hz)
#     points = point_cloud2.read_points_list(msg,field_names=("x","y","z"))
#     point_cloud_numpy = np.array(points) #(length,3)
#     point_cloud_queue.append(point_cloud_numpy)
#     print(f"pointcloud : {len(point_cloud_queue)}")
#     rate.sleep()

def imu_callback(msg):
    rate = rospy.Rate(hz)
    orientation = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
    angular_velocity = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
    linear_acceleration = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
    # 将数据存储为 NumPy 数组
    imu_data = np.array(orientation + angular_velocity + linear_acceleration)
    imu_queue.append(imu_data)
    print(f"imu : {len(imu_queue)}")
    rate.sleep()

def motor_callback(msg):
    rate = rospy.Rate(hz)
    motor = [msg.motor.speed1, msg.motor.speed2]
    # 将数据存储为 NumPy 数组
    motor_data = np.array(motor)
    motor_queue.append(motor_data)
    print(f"motor : {len(motor_queue)}")
    rate.sleep()

# def image_leg_callback(msg):
#     rate = rospy.Rate(hz)
#     bridge = CvBridge()
#     cv_image = bridge.compressed_imgmsg_to_cv2(msg, "rgb8")
#     np_image = np.array(cv_image)
#     np_image = np_image.transpose(2, 0, 1)
#     image_leg_queue.append(np_image)
#     print(f"leg : {len(image_leg_queue)}")
#     rate.sleep()


if __name__ == "__main__":
    rospy.init_node("multithreaded_ros_subscriber")


    rospy.Subscriber("/usb_cam0/image_raw0/compressed", CompressedImage, image1_callback)
    rospy.Subscriber("/usb_cam2/image_raw2/compressed", CompressedImage, image2_callback)
    rospy.Subscriber("/usb_cam4/image_raw4/compressed", CompressedImage, image3_callback)
    rospy.Subscriber("/usb_cam6/image_raw6/compressed", CompressedImage, image4_callback)
    rospy.Subscriber("/points_raw", PointCloud2, pointcloud_callback)
    rospy.Subscriber("/imu", Imu, imu_callback)
    rospy.Subscriber("/ecparm", ECParm, motor_callback)
    rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, image_leg_callback)

    rospy.spin()
