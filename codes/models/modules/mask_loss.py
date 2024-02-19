import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class mask_loss:
    def __init__(self, opt, device=None):
        loss_type_mask = opt['mask_criterion']
        if loss_type_mask == 'l1':
            self.cri_mask = nn.L1Loss().to(device)
        elif loss_type_mask == 'l2':
            self.cri_mask = nn.MSELoss().to(device)
        elif loss_type_mask == 'cb':
            self.cri_mask = CharbonnierLoss().to(device)
        elif loss_type_mask == 'smoothl1':
            self.cri_mask = nn.SmoothL1Loss(reduction='none').to(device)
        else:
            raise NotImplementedError('Loss type [{:s}] for depth loss is not recognized.'.format(loss_type_mask))
        self.loss_type_mask = loss_type_mask
        self.l_mask_w = opt['mask_weight']
    
    def get_mask_loss(self, sr_img, hr_img, depthMaskList):
        bs, ch, h, w = depthMaskList.shape
        randn = np.random.randint(0, ch, 1)
        randn = randn[0] 
        depthMask = depthMaskList[:,randn,:,:]
        depthMask = depthMask.unsqueeze(1)
        resized_depthMask =  F.interpolate(depthMask, size=sr_img.size()[2:], mode='nearest')
        resized_depthMask = torch.cat([resized_depthMask,resized_depthMask,resized_depthMask], dim=1)

        masked_sr_img = torch.mul(resized_depthMask, sr_img)
        masked_hr_img = torch.mul(resized_depthMask, hr_img)
        
        if self.loss_type_mask == 'smoothl1':
            loss = self.cri_mask(masked_sr_img, masked_hr_img)
            loss = torch.sum(loss)
            total = torch.sum(resized_depthMask)
            loss = loss / total * self.l_mask_w
            return loss
        loss = self.l_mask_w * self.cri_mask(masked_sr_img, masked_hr_img)
        return loss


class dynamic_weight_mask_loss(nn.Module):
    def __init__(self, opt, device=None, num_trainable_para=10):
        super(dynamic_weight_mask_loss, self).__init__()
        loss_type_mask = opt['dynamic_criterion']
        if loss_type_mask == 'l1':
            self.cri_mask = nn.L1Loss().to(device)
        elif loss_type_mask == 'l2':
            self.cri_mask = nn.MSELoss().to(device)
        elif loss_type_mask == 'cb':
            self.cri_mask = CharbonnierLoss().to(device)
        elif loss_type_mask == 'smoothl1':
            self.cri_mask = nn.SmoothL1Loss(reduction='none').to(device)
        else:
            raise NotImplementedError('Loss type [{:s}] for depth loss is not recognized.'.format(loss_type_mask))
        self.loss_type_mask = loss_type_mask
        self.l_mask_w = opt['dynamic_weight']
        # trainable hyper-parameters
        self.num_trainable_parameters = num_trainable_para
        self.trainable_weight = nn.Parameter(torch.ones(num_trainable_para))
    
    def forward(self, sr_img, hr_img, depthMaskList):
        bs, ch, h, w = depthMaskList.shape
        assert self.num_trainable_parameters == ch, "The number of trainable parameters for dynamic loss is not enought."
        loss_list = []
        weighted_loss_list = []
        softmax_weight = F.softmax(self.trainable_weight, dim = 0)
        for i in range(ch):
            depthMask = depthMaskList[:,i,:,:]
            depthMask = depthMask.unsqueeze(1)
            resized_depthMask =  F.interpolate(depthMask, size=sr_img.size()[2:], mode='nearest')
            resized_depthMask = torch.cat([resized_depthMask,resized_depthMask,resized_depthMask], dim=1)

            masked_sr_img = torch.mul(resized_depthMask, sr_img)
            masked_hr_img = torch.mul(resized_depthMask, hr_img)
            
            if self.loss_type_mask == 'smoothl1':
                loss = self.cri_mask(masked_sr_img, masked_hr_img)
                loss = torch.sum(loss)
                total = torch.sum(resized_depthMask)
                loss = loss / total
            else:
                loss = self.cri_mask(masked_sr_img, masked_hr_img)
            loss_list.append(loss)
            weighted_loss_list.append(softmax_weight[i]*loss)
        weighted_loss = sum(weighted_loss_list) * self.l_mask_w

        return loss_list, weighted_loss_list, weighted_loss, softmax_weight 


