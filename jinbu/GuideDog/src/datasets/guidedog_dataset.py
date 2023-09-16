"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""
import base64
import code
import json
import math
import os
import os.path as op
import pickle
import random
import time
# Ignore warnings
import warnings
from collections import Counter

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from numpy.random import randint
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import io, transforms

from src.utils.load_files import (find_file_path_in_yaml,
                                  load_box_linelist_file, load_from_yaml_file,
                                  load_linelist_file)
from src.utils.logger import LOGGER
from src.utils.tsv_file import CompositeTSVFile, TSVFile
from src.utils.tsv_file_ops import tsv_reader

from .data_utils.image_ops import img_from_base64
from .data_utils.video_bbox_transforms import video_bbox_prcoess
from .data_utils.video_ops import (extract_frames_from_video_binary,
                                   extract_frames_from_video_path)
# video_transforms & volume_transforms from https://github.com/hassony2/torch_videovision
from .data_utils.video_transforms import (CenterCrop, ColorJitter, Compose,
                                          Normalize, RandomCrop,
                                          RandomHorizontalFlip,
                                          RandomResizedCrop, Resize)
from .data_utils.volume_transforms import ClipToTensor

from .guidedog_dataset_help import DogDataset

class My_Dog_DATASET(object):
    def __init__(self, args, is_train=True, mode='train'):

        self.args = args
        self.mode = mode

        with open("dataset/data_split.json", "r") as f:
            self.data_index = json.load(f)[mode]

        self.raw_dataset = DogDataset()

        self.detr_aug = True

        self.is_composite = False

        self.is_train = is_train
        self.img_res = getattr(args, 'img_res', 224)
        self.patch_size = getattr(args, 'patch_size', 16)

        self.img_feature_dim = args.img_feature_dim
        self.decoder_num_frames = getattr(args, 'max_num_frames', 2)

        LOGGER.info(f'isTrainData: {self.is_train}\n[PyAV video parameters] '
                    f'Num of Frame: {self.decoder_num_frames}, ')

        if is_train==True:
            self.raw_video_crop_list = [
                Resize(self.img_res),
                Resize((self.img_res,self.img_res)),
                ClipToTensor(channel_nb=3),
                Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]
        else:
            self.raw_video_crop_list = [
                Resize(self.img_res),
                Resize((self.img_res,self.img_res)),
                ClipToTensor(channel_nb=3),
                Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]
        
        self.raw_video_prcoess = Compose(self.raw_video_crop_list)
        self.video_bbox_process = video_bbox_prcoess(is_train, self.img_res)


    def apply_augmentations(self, frames):

        # TODO: this should be changed when the levels are changed
        frames_num_to_return = self.decoder_num_frames

        # if failed to decode video, generate fake frames (should be corner case)
        if frames is None:
            frames = np.zeros((frames_num_to_return,self.img_res,self.img_res,3)).astype(np.uint8)
        # (T, C, H, W) -> (T, H, W, C), channel is RGB
        elif 'torch' in str(frames.dtype):
            frames = frames.numpy()
            frames = np.transpose(frames, (0, 2, 3, 1))
        else:
            frames = frames.astype(np.uint8)
            frames = np.transpose(frames, (0, 2, 3, 1))
        num_of_frames, height, width, channels = frames.shape

        frame_list = []
        for i in range(frames_num_to_return):
            if num_of_frames==1: 
                # if it is from image-caption dataset, we duplicate the image
                # convert numpy to PIL format, compatible to augmentation operations
                frame_list.append(Image.fromarray(frames[0]))
            else:
                # if it is from video-caption dataset, we add each frame to the list
                # convert numpy to PIL format, compatible to augmentation operations
                frame_list.append(Image.fromarray(frames[i]))
        
        # adapt from torch_videovision: https://github.com/hassony2/torch_videovision
        # after augmentation, output tensor (C x T x H x W) in the range [0, 1.0]
        crop_frames = self.raw_video_prcoess(frame_list)
        # (C x T x H x W) --> (T x C x H x W)
        crop_frames = crop_frames.permute(1, 0, 2, 3)
        return crop_frames 
    
    def apply_bbox_augmentations(self, frames):

        # TODO: this should be changed when the levels are changed
        frames_num_to_return = self.decoder_num_frames

        # if failed to decode video, generate fake frames (should be corner case)
        if frames is None:
            frames = np.zeros((frames_num_to_return,self.img_res,self.img_res,3)).astype(np.uint8)
        # (T, C, H, W) -> (T, H, W, C), channel is RGB
        elif 'torch' in str(frames.dtype):
            frames = frames.numpy()
            frames = np.transpose(frames, (0, 2, 3, 1))
        else:
            frames = frames.astype(np.uint8)
            frames = np.transpose(frames, (0, 2, 3, 1))
        num_of_frames, height, width, channels = frames.shape

        frame_list = []
        for i in range(frames_num_to_return):
            if num_of_frames==1: 
                # if it is from image-caption dataset, we duplicate the image
                # convert numpy to PIL format, compatible to augmentation operations
                frame_list.append(Image.fromarray(frames[0]))
            else:
                # if it is from video-caption dataset, we add each frame to the list
                # convert numpy to PIL format, compatible to augmentation operations
                frame_list.append(Image.fromarray(frames[i]))
        
        # adapt from torch_videovision: https://github.com/hassony2/torch_videovision
        # after augmentation, output tensor (C x T x H x W) in the range [0, 1.0]
        crop_frames, bbox = self.video_bbox_process(frame_list)

        # (C x T x H x W) --> (T x C x H x W)
        crop_frames = crop_frames.permute(1, 0, 2, 3)
        return crop_frames, bbox


    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        image_key = str(self.data_index[idx]).zfill(6)
        sample_data = self.raw_dataset.__getitem__(self.data_index[idx])

        if True:
            # TODO: pred the first label
            class_label = torch.tensor(sample_data['label'][0])

            # find the video path in our machine
            imgs_path = sample_data['video']
            assert (not imgs_path[-1].startswith('/')) and (imgs_path[-1].endswith('.png') or imgs_path[-1].endswith('.jpg'))

            def uniform_sampling(lst, k):
                # must inculde the last one
                if k == 1:
                    return lst[-1]

                interval = (len(lst) - 1) / (k - 1)

                sample_indices = [int(i * interval) for i in range(k - 1)]
                sample_indices.append(len(lst) - 1)
                sample = [lst[i] for i in sample_indices]
                return sample

            if len(imgs_path) == 1:
                raw_frames = np.array([cv2.resize(cv2.imread(imgs_path[-1]),(1024, 1024))[:,:,::-1]]*self.decoder_num_frames)
            elif len(imgs_path) > 1:
                imgs_path = uniform_sampling(imgs_path, self.decoder_num_frames)
                raw_frames = np.array([cv2.resize(cv2.imread(imgs_path[i]),(1024, 1024))[:,:,::-1] for i in range(len(imgs_path))])
            else:
                raise Exception(f"the frame number in a video {imgs_path} is less than 1")

            # (T H W C) to (T C H W)
            raw_frames = (np.transpose(raw_frames, (0, 3, 1, 2)))

        # apply augmentation. frozen-in-time if the input is an image
        # preproc_frames: (T, C, H, W), C = 3, H = W = self.img_res, channel is RGB   
        if self.detr_aug:
            preproc_raw_frames, _ = self.apply_bbox_augmentations(raw_frames)
        else:
            preproc_raw_frames = self.apply_augmentations(raw_frames)

        # preparing outputs
        meta_data = {}
        meta_data['raw_image'] = raw_frames[0]

        example =  (preproc_raw_frames, class_label)
        # return image_key, example, meta_data
        return image_key, example, meta_data

