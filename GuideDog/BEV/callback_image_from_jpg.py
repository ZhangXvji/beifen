#!/usr/bin/env python


import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
import cv2

def callback_image(path):
     pass
     cv_image = np.array(cv2.imread(path))

     # # 将OpenCV格式图片转Numpy数组
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
                   new_mask[r.masks.data[i].cpu().numpy()[..., 0] != 0] = r.boxes.cls[i].cpu().numpy()+1
                   mask_result = mask_result + new_mask
     return(mask_result)


          