from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import cv2

from .mono_dataset import MonoDataset


class EndovisDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(EndovisDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.82, 0, 0.5, 0],
                           [0, 1.02, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.side_map = {"1": 1, "2": 2, "l": 1, "r": 2}

    def check_depth(self):
        return True #False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        color = self.center_crop(color)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    
    def center_crop(self, color):
        #  center crop image
        # print("color.size:",color.size)
        width, height = color.size   # Get dimensions
        new_width = 320
        new_height = 256 #320
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2

        # Crop the center of the image
        color = color.crop((left, top, right, bottom))
        return color
    
    def center_crop_depth(self, color):
        #  center crop image
        # print("color.size:",color.size)
        height, width = color.shape   # Get dimensions
        new_width = 320
        new_height = 256 #320
        left = int((width - new_width)/2)
        top = int((height - new_height)/2)
        right = int((width + new_width)/2)
        bottom = int((height + new_height)/2)

        # Crop the center of the image
        color = color[top:bottom, left:right]
        return color


class EndovisRAWDataset(EndovisDataset):
    def __init__(self, *args, **kwargs):
        super(EndovisRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        # /apdcephfs/share_916081/jarviswang/wt/dataset/endoscopy/Hamlyn/rectified01/image01/00 00 00 06 67
        # f_str = "image_{}{}".format(frame_index, self.img_ext)
        f_str = "{}{}".format(str(frame_index).zfill(10), '.jpg') #self.img_ext)
        # print(folder, frame_index, side)
        # print(f_str)
        image_path = os.path.join(self.data_path, folder, "image0{}".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        # f_str = "scene_points{:06d}.tiff".format(frame_index-1)
        # 00 00 00 00 00.png
        f_str = "{}{}".format(str(frame_index).zfill(10), '.png') #self.img_ext)

        depth_path = os.path.join(
            self.data_path,
            folder,
            "depth0{}".format(self.side_map[side]),
            f_str)
        # print("depth_path:",depth_path)
        depth_gt = cv2.imread(depth_path, 3)
        depth_gt = depth_gt[:, :, 0]
        # print("depth_gt:",depth_gt.shape)
        # depth_gt = depth_gt[0:1024, :]
        # center crop
        depth_gt = self.center_crop_depth(depth_gt)

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
