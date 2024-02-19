import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import sys
import os
import copy
from PIL import Image
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.util import imresize_np
    from utils import util as utils
except ImportError:
    pass


class EndoSceneDepthnewDataset(data.Dataset):
    '''
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(EndoSceneDepthnewDataset, self).__init__()
        self.opt = opt
        self.opt_F = opt
        self.opt_P = opt
        self.opt_C = opt
        self.LR_paths, self.GT_paths = [], []
        self.Depth_paths = []
        self.LR_env, self.GT_env = None, None  # environment for lmdb
        self.LR_size, self.GT_size = opt['LR_size'], opt['GT_size']
        self.isMonoDepth = False #"depth_old_model" in opt['dataroot_depthMap']
        self.use_seg_label = opt['use_seg_label']
        self.num_classes = opt['num_classes']

        # read data list
        filelist = open(opt['dataset_split_list'], 'r')
        imglist = filelist.readlines()
        filelist.close()

        # read image list
        for i in range(len(imglist)):
            imglist[i] = imglist[i].strip()
            imgname = imglist[i].split('.tif')[0] + '.png'
            # self.LR_paths.append(os.path.join(opt['dataroot_LQ'], 'x' + str(opt['scale']), imglist[i]))
            # self.GT_paths.append(os.path.join(opt['dataroot_GT'], imglist[i]))
            self.LR_paths.append(os.path.join(opt['dataroot_LQ'], imgname))
            self.GT_paths.append(os.path.join(opt['dataroot_GT'], imgname))
            depthFile = imglist[i].split('.')[0] + '_disp.npy'
            # self.Depth_paths.append(os.path.join(opt['dataroot_depthMap'], 'x' + str(opt['scale']) + '_npy', depthFile)) 
            self.Depth_paths.append(os.path.join(opt['dataroot_depthMap'], 'x2_npy', depthFile)) 

        # obtain the segmentation label
        if self.use_seg_label:
            self.seg_label_list = []
            for i in range(len(imglist)):
                imglist[i] = imglist[i].strip()
                self.seg_label_list.append(os.path.join(opt['dataroot_label'], imglist[i]))

        # read image list from lmdb or image files
        # if opt['phase'] == 'train' and opt['data_augment']:
        #     aug_paths = [] 
        #     origin_LR_paths = copy.deepcopy(self.LR_paths)
        #     for imagepath in self.LR_paths:
        #         imgname = imagepath.split('/')[-1]
        #         imgname = imgname.split('.')[0] + '_DA.jpg'
        #         aug_paths.append(os.path.join(opt['dataroot_LQ_Aug'], imgname)
        #     self.LR_paths.extend(aug_paths)
        #     self.GT_paths.extend(self.GT_paths)


        assert self.GT_paths, 'Error: GT paths are empty.'
        if self.LR_paths and self.GT_paths and self.Depth_paths:
            assert len(self.LR_paths) == len(self.GT_paths) == len(self.Depth_paths) , 'GT, LR and Depth datasets have different number of images - {}, {}, {}.'.format(len(self.LR_paths), len(self.GT_paths), len(self.Depth_paths))
        self.random_scale_list = [1]


    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LR_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        if self.opt['data_type'] == 'lmdb':
            if (self.GT_env is None) or (self.LR_env is None):
                self._init_lmdb()

        GT_path, LR_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        LR_size = self.opt['LR_size']


        # get GT image
        GT_path = self.GT_paths[index]
        if self.opt['data_type'] == 'lmdb':
            resolution = [int(s) for s in self.GT_sizes[index].split('_')]
        else:
            resolution = None
        img_GT = util.read_img(self.GT_env, GT_path, resolution) #return: Numpy float32, HWC, BGR, [0,1]

        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_GT = util.modcrop(img_GT, scale)


        # get LR image
        if self.LR_paths: # LR exist
            LR_path = self.LR_paths[index]
            if self.opt['data_type'] == 'lmdb':
                resolution = [int(s) for s in self.LR_sizes[index].split('_')]
            else:
                resolution = None
            img_LR = util.read_img(self.LR_env, LR_path, resolution)
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                # force to 3 channels
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LR = util.imresize_np(img_GT, 1 / scale, True)
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)

        # get depth map
        depth_map = np.load(self.Depth_paths[index])
        ### resize
        # print("img_LR:",img_LR.shape)
        depth_h, depth_w, _ = img_LR.shape
        depth_map = cv2.resize(depth_map,(depth_w,depth_h),cv2.INTER_CUBIC)

        if self.isMonoDepth:
            depth_map = depth_map.squeeze(1)
        depth_map = torch.from_numpy(depth_map).float()
        
        # get depth mask
        depth_maskList = self.getDepthMask(depth_map,self.opt['depthFixedRange'],self.opt['depthMaskNum'])
        # to numpy
        depth_map = depth_map.numpy()
        if self.isMonoDepth:
            depth_map = depth_map.transpose(1,2,0)
        else:
            depth_map = np.expand_dims(depth_map, axis=2)
        depth_maskList = depth_maskList.numpy()
        depth_maskList = depth_maskList.transpose(1,2,0)

        # get label 
        if self.use_seg_label:
            label_path = self.seg_label_list[index]
            segmentation = np.array(Image.open(label_path))
            (T, segmentation) = cv2.threshold(segmentation, 0, 255, cv2.THRESH_BINARY)
            segmentation = segmentation/255.
            segmentation = segmentation.reshape([segmentation.shape[0], segmentation.shape[1],1])
            

        if self.opt['phase'] == 'train':
            H, W, C = img_LR.shape
            assert LR_size == GT_size // scale, 'GT size does not match LR size'

            # randomly crop
            # rnd_h = random.randint(0, max(0, H - LR_size))
            # rnd_w = random.randint(0, max(0, W - LR_size))
            # img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            # rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            # img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            if self.use_seg_label:
                img_LR, img_GT, depth_map, depth_maskList, segmentation = util.augment([img_LR, img_GT, depth_map, depth_maskList, segmentation], self.opt['use_flip'], self.opt['use_rot'], self.opt['mode'])
            else:
                img_LR, img_GT, depth_map, depth_maskList = util.augment([img_LR, img_GT, depth_map, depth_maskList], self.opt['use_flip'], self.opt['use_rot'], self.opt['mode'])


        # change color space if necessary
        if self.opt['color']:
            H, W, C = img_LR.shape
            img_LR = util.channel_convert(C, self.opt['color'], [img_LR])[0]  # TODO during val no definition
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()
        depth_map = torch.from_numpy(np.ascontiguousarray(np.transpose(depth_map, (2, 0, 1)))).float()
        depth_maskList = torch.from_numpy(np.ascontiguousarray(np.transpose(depth_maskList, (2, 0, 1)))).float()

        if self.use_seg_label:
            segmentation = segmentation.reshape([segmentation.shape[0],segmentation.shape[1]])
            segmentation = torch.from_numpy(np.ascontiguousarray(segmentation)).long()
            segmentation_onehot = util.get_one_hot(segmentation, self.num_classes)
            segmentation_onehot = segmentation_onehot.permute(2,0,1).float()

        if LR_path is None:
            LR_path = GT_path

        ret_dict = {'LQ': img_LR, 'GT': img_GT, 'LQ_path': LR_path, 'GT_path': GT_path, 'Depth':depth_map, 'DepthMaskList':depth_maskList}
        if self.use_seg_label:
            ret_dict['Seg'] = segmentation
            ret_dict['Seg_onehot'] = segmentation_onehot
        return ret_dict

    def __len__(self):
        return len(self.GT_paths)
    
    def getDepthMask(self, depthMap,depthFixedRange=True,depthMaskNum=10):
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
        out = torch.cat(depthMaskList,0)
        return out