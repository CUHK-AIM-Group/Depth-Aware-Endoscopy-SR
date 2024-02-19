import random
import numpy as np
import lmdb
import torch
import torch.utils.data as data
import data.util as util


class LQDataset(data.Dataset):
    '''Read LR images to Predictor.'''

    def __init__(self, opt):
        super(LQDataset, self).__init__()
        self.opt = opt
        self.opt_P = opt
        self.LR_paths = None
        self.LR_sizes = None  # environment for lmdb
        self.LR_env = None
        self.LR_size = opt['LR_size']

        self.real_ker_path = '/mnt/yjchai/SR_data/Flickr2K/kermap.pt'
        self.real_ker_map_list = util.load_ker_map_list(self.real_ker_path)

        # read image list from lmdb or image files
        if opt['data_type'] == 'lmdb':
            self.LR_paths, self.LR_sizes = util.get_image_paths(opt['data_type'], opt['dataroot_LQ'])
        elif opt['data_type'] == 'img':
            self.LR_paths = util.get_image_paths(opt['data_type'], opt['dataroot_LQ']) #LR_list
        else:
            print('Error: data_type is not matched in Dataset')
        assert self.LR_paths, 'Error: LR paths are empty.'



    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.LR_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        if self.opt['data_type'] == 'lmdb':
            if self.LR_env is None:
                self._init_lmdb()

        LR_size = self.LR_size

        # get real kernel map
        real_ker_map = self.real_ker_map_list[index].float()

        # get LR image
        LR_path = self.LR_paths[index]
        if self.opt['data_type'] == 'lmdb':
            resolution = [int(s) for s in self.LR_sizes[index].split('_')]
        else:
            resolution = None
        img_LR = util.read_img(self.LR_env, LR_path, resolution)
        H, W, C = img_LR.shape

        if self.opt['phase'] == 'train':
            #randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]

            # augmentation - flip, rotate
            img_LR = util.augment(img_LR, self.opt['use_flip'], self.opt['use_rot'], self.opt['mode'])

        # change color space if necessary
        if self.opt['color']:
            img_LR = util.channel_convert(C, self.opt['color'], [img_LR])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_LR.shape[2] == 3:
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        return {'LQ': img_LR, 'LQ_path': LR_path, 'real_ker': real_ker_map}

    def __len__(self):
        return len(self.LR_paths)
