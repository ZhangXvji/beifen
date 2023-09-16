#!/usr/bin/env python

from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from ultralytics import YOLO

def callback_image(image_msg):
     pass
     # 将ROS图像消息转换为OpenCV格式
     bridge = CvBridge()
     cv_image = bridge.compressed_imgmsg_to_cv2(image_msg, "bgr8")

     # # 将OpenCV格式图片转Numpy数组
     np_image = np.array(cv_image)
     
     height, width, _ = np_image.shape
     result_image = np.zeros((height, width, 3), dtype=np.uint8)
     
     model = YOLO('yolov8n-seg.pt')

     results = model(np_image)

     boxes = results[0].boxes
     masks = results[0].masks
     mask_result = np.zeros((height, width, 1), dtype=np.uint8)
     if(len(boxes.cls)!=0):
          for i in range(len(boxes.cls)):
               new_mask = np.zeros((height, width, 1), dtype=np.uint8)
               new_mask[masks.data[i].cpu().numpy()[..., 0] != 0] = boxes.cls[i].cpu().numpy()
               mask_result = mask_result + new_mask
               
               cv2.imshow("origin", results[0].plot())
               cv2.imshow("result", mask_result)
               cv2.waitKey(100)