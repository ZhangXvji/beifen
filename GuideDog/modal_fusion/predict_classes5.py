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
import threading
from cv_bridge import CvBridge
from queue import Queue
from std_msgs.msg import Int8, Float32
import time
import socket
import struct

pub = rospy.Publisher('/output_class', Float32, queue_size=10)

class Args:
    def __init__(self) -> None:
        # 省略训练相关的参数，只保留与预测相关的参数
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = "./model_epoch_24.pth"
        self.check_frame = 8 # 理论上1秒预测1次，但是有些冗余时间更好，设置为0.8秒预测1次
args = Args()

hz = 10
max_length = 60

img2_queue = deque(maxlen=max_length)
imgL_queue = deque(maxlen=max_length)
imu_queue = deque(maxlen=max_length)
motor_queue = deque(maxlen=max_length)

# 初始化计数器
prev_queue_lengths = {
    'img2': 0,
    'imgL': 0,
    'imu': 0,
    'motor': 0
}
count_if_10_frame = 0

# stop_orentiation = [False, False, False, False]
stop_orentiation = [True, True, True, True]
stop_lock = threading.Lock()

def predict(sample_batch):
    global stop_orentiation
    # 加载预训练模型
    model = torch.load(args.model_path)
    # 设置模型为评估模式
    with torch.no_grad():
        outputs = model(sample_batch['video2'].to(args.device),sample_batch['videoL'].to(args.device),sample_batch['imu'].to(args.device), sample_batch['motor'].to(args.device))
        print(outputs)
        if not stop_orentiation[0]:
            outputs[0, 1] = 0.0

        if not stop_orentiation[1]:
            outputs[0, 2] = 0.0

        if not stop_orentiation[2]:
            outputs[0, 4] = 0.0

        if not stop_orentiation[3]:
            outputs[0, 3] = 0.0


        # 处理模型输出，这里假设您要获得每个样本的类别预测
        predicted_classes = torch.argmax(outputs[:, 1:], dim=1).cpu().numpy() + 1.0
        if torch.sum(outputs[:, 1:]) == 0:
            predicted_classes = 0.0

        return predicted_classes

def sample():
    global img2_queue, imgL_queue, imu_queue, motor_queue

    video2 = get_video_tensor(img2_queue)
    videoL = get_video_tensor(imgL_queue)
    imu = get_imu_tensor(imu_queue)
    motor = get_motor_tensor(motor_queue)

    sample = {
                'video2': video2.float(),
                'videoL': videoL.float(),
                'imu': imu.float(),
                'motor': motor.float()}
    return sample

def get_video_tensor(image_tensors):
    # 将图像张量堆叠成一个批次
    image_list = list(image_tensors)
    # 将图像张量堆叠成一个批次
    image_batch = torch.stack(image_list)  # 60x56x56x3 DHWC
    image_batch = torch.permute(image_batch, (3, 0, 1, 2))  # 3,60,56,56
    tensor_data = torch.tensor(image_batch, dtype=torch.float32).unsqueeze(0)
    # print(tensor_data.shape)
    return tensor_data

def get_imu_tensor(imu_tensors):
    imu_list = list(imu_tensors)
    imu_batch = torch.stack(imu_list)
    tensor_data = torch.tensor(imu_batch, dtype=torch.float32).unsqueeze(0)
    tensor_data = torch.tensor(tensor_data, dtype=torch.float32).unsqueeze(0)
    # print(tensor_data.shape)
    return tensor_data
def get_motor_tensor(motor_tensors):
    motor_list = list(motor_tensors)
    motor_batch = torch.stack(motor_list)
    tensor_data = torch.tensor(motor_batch, dtype=torch.float32).unsqueeze(0)
    tensor_data = torch.tensor(tensor_data, dtype=torch.float32).unsqueeze(0)
    # print(tensor_data.shape)
    return tensor_data


def check_queues_thread_func():
    rate = rospy.Rate(hz*2)
    while not rospy.is_shutdown():
        check_queues()
        rate.sleep()

def check_queues():
    global prev_queue_lengths, count_if_10_frame, args, pub
    # 在每个回调函数中检查队列是否有新数据
    # 如果队列加入了新数据，增加计数器
    if all(value == 1 for value in prev_queue_lengths.values()):
        count_if_10_frame = count_if_10_frame+1
        for key in prev_queue_lengths:
            prev_queue_lengths[key] = 0
            if(count_if_10_frame == 100):
                data = sample()
                output_class = predict(data)
                pub.publish(output_class)
                count_if_10_frame = count_if_10_frame - args.check_frame

def image2_callback(msg):
    global img2_queue,prev_queue_lengths
    rate = rospy.Rate(hz)

    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
    cv_image = cv2.resize(cv_image, (56, 56))
    np_image = np.array(cv_image) / 255.0
    tensor = torch.tensor(np_image, dtype=torch.float32)
    # print(tensor.shape)
    img2_queue.append(tensor)
    prev_queue_lengths["img2"] = 1
    rate.sleep()

def imageL_callback(msg):
    global imgL_queue, prev_queue_lengths
    rate = rospy.Rate(hz)

    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
    cv_image = cv2.resize(cv_image, (56, 56))
    np_image = np.array(cv_image) / 255.0
    tensor = torch.tensor(np_image, dtype=torch.float32)
    # print(tensor.shape)
    imgL_queue.append(tensor)
    prev_queue_lengths["imgL"] = 1
    rate.sleep()

def imu_callback(msg):
    global imu_queue, prev_queue_lengths
    rate = rospy.Rate(hz)
    orientation = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
    angular_velocity = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
    linear_acceleration = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
    # 将数据存储为 NumPy 数组
    imu_data = np.array(orientation + angular_velocity + linear_acceleration)
    imu_data = torch.tensor(imu_data)
    imu_queue.append(imu_data)
    # print(f"imu : {len(imu_queue)}")
    prev_queue_lengths["imu"] = 1
    rate.sleep()

def motor_callback(msg):
    global motor_queue, prev_queue_lengths
    rate = rospy.Rate(hz)
    motor = [msg.motor.speed1, msg.motor.speed2]
    # 将数据存储为 NumPy 数组
    motor_data = np.array(motor)
    motor_data = torch.tensor(motor_data)
    motor_queue.append(motor_data)
    # print(f"motor : {len(motor_queue)}")
    prev_queue_lengths["motor"] = 1
    rate.sleep()

def udp():
    global stop_orentiation
    udp_addr = ("", 8063)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(udp_addr)
    while True:
        print("happy")
        data, addr = udp_socket.recvfrom(1024)
        new_data = struct.unpack("????", data)
        stop_orentiation = new_data
        print(stop_orentiation)
        time.sleep(0.01)

if __name__ == "__main__":
    rospy.init_node("multithreaded_ros_subscriber")
    check_queues_thread = threading.Thread(target=check_queues_thread_func)
    udp_thread = threading.Thread(target=udp)
    check_queues_thread.start()
    udp_thread.start()
    rospy.Subscriber("/usb_cam2/image_raw2", Image, image2_callback)
    rospy.Subscriber("/usb_cam/image_raw", Image, imageL_callback)
    rospy.Subscriber("/imu/data", Imu, imu_callback)
    rospy.Subscriber("/ecparm", ECParm, motor_callback)
    check_queues_thread.join()
    udp_thread.join()
    rospy.spin()

