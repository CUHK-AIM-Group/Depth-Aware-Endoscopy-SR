import torch.nn as nn
from models.vgg_extractor.vggnet import VGGNet

class vgg_depth_loss:
    def __init__(self, opt, device=None):
        self.vggnet = VGGNet(opt['vgg_type'], opt['vgg_layers'], device=device)
        self.vggnet.vgg.to(device)
        self.use_depth_mask = opt['use_depth_mask']
        self.use_depth_mask_layer = opt['use_depth_mask_layer']
        self.vgg_weight = opt['vgg_weight']

        # criterion

        loss_type_vgg = opt['vgg_criterion']
        if loss_type_vgg == 'l1':
            self.cri_vgg = nn.L1Loss().to(device)
        elif loss_type_vgg == 'l2':
            self.cri_vgg = nn.MSELoss().to(device)
        elif loss_type_vgg == 'cb':
            self.cri_vgg = CharbonnierLoss().to(device)
        else:
            raise NotImplementedError('Loss type [{:s}] for vgg loss is not recognized.'.format(loss_type_vgg))

    def get_loss(self, preds, targs, depth_mask=None):
        preds_feat = self.vggnet(preds)
        targs_feat = self.vggnet(targs)
        
        # multiply the feature with the depth mask
        # if self.use_depth_mask:
        N = len(preds_feat)
        total_loss = 0
        loss_list = []
        for i in range(N):
            loss = self.vgg_weight[i] * self.cri_vgg(preds_feat[i], targs_feat[i]) 
            loss_list.append(loss)
            total_loss += loss
        return total_loss, loss_list