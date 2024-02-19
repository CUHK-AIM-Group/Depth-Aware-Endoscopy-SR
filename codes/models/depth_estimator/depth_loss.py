import os
import numpy as np
import torch
import torch.nn as nn
import pandas
from models.depth_estimator.depth_decoder import DepthDecoder
from models.depth_estimator.resnet_encoder import ResnetEncoder

class depth_estimator:
    def __init__(self, opt, device=None):
        loss_type_depth = opt['depth_criterion']
        if loss_type_depth == 'l1':
            self.cri_depth = nn.L1Loss().to(device)
        elif loss_type_depth == 'l2':
            self.cri_depth = nn.MSELoss().to(device)
        elif loss_type_depth == 'cb':
            self.cri_depth = CharbonnierLoss().to(device)
        else:
            raise NotImplementedError('Loss type [{:s}] for depth loss is not recognized.'.format(loss_type_depth))
        self.l_depth_w = opt['depth_weight']

        # initialize models
        model_path = opt['pretrained_model_path']
        print("-> Loading depth estimator model from ", model_path)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        # LOADING PRETRAINED MODEL
        print("   Loading pretrained encoder")
        self.encoder = ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=device)
        
        # extract the height and width of image that this model was trained with
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(device)
        self.encoder.eval()

        print("   Loading pretrained decoder")
        self.depth_decoder = DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        self.depth_decoder.load_state_dict(loaded_dict)

        self.depth_decoder.to(device)
        self.depth_decoder.eval()

    def get_depth_feature(self, image):
        # resize to the feed_width and feed_height
        resized_image = torch.nn.functional.interpolate(
            image, (self.feed_height, self.feed_width), mode="bilinear", align_corners=False)

        # switch channel: BGR to RGB
        # resized_image = resized_image[:, [2,1,0], :, :]

        # PREDICTION
        features = self.encoder(resized_image)
        outputs = self.depth_decoder(features)

        disp = []
        for i in range(4):
            disp.append(outputs[("disp", i)])
        return disp

    def calculate_depth_loss(self, sr_img, hr_img):
        sr_disp = self.get_depth_feature(sr_img)
        hr_disp = self.get_depth_feature(hr_img)

        loss = []
        for i in range(4):
            loss.append(self.l_depth_w[i] * self.cri_depth(sr_disp[i], hr_disp[i]))
        total_loss = sum(loss)
        return total_loss, loss, sr_disp, hr_disp