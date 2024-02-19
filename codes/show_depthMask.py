import torch
import numpy as np
import cv2
import os
def getDepthMask(depthMap,depthFixedRange=True,depthMaskNum=10):
    depthMap = torch.squeeze(depthMap)
    max_val = torch.max(depthMap)
    min_val = torch.min(depthMap)
    if depthFixedRange :
        max_val = 1
        min_val = 0 
    interval = (max_val-min_val) / depthMaskNum
    maskRangeList =[]
    depthMaskList = []
    for i in range(depthMaskNum):
        start_val = min_val + interval*i
        end_val = min_val + interval*(i+1)
        maskRangeList.append([start_val,end_val])
    for maskRange in maskRangeList:
        start_v = maskRange[0]
        end_v = maskRange[1]
        mask = torch.zeros(depthMap.shape)
        mask[(depthMap >= start_v) & (depthMap < end_v)] = 1
        mask = torch.unsqueeze(mask,0)
        depthMaskList.append(mask)
    return depthMaskList


depthRoot = '/home/wentchen/dataset/Kvasir_dataset/x8_128_depth_map/test_npy'
depth_map_x8 = np.load(os.path.join(depthRoot,'kvasir_v2__fold_9_train_class_normal-z-line_frame_404_disp.npy'))
depth_map_x8 = depth_map_x8.squeeze(1)
depth_map_x8 = torch.from_numpy(depth_map_x8).float()

outlist = getDepthMask(depth_map_x8,depthFixedRange=False,depthMaskNum=10)
for i in range(len(outlist)):
    cv2.imwrite(str(i+1)+'.png',outlist[i].numpy().squeeze()*255)