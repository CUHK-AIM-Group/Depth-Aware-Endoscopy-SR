import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import CharbonnierLoss,SSIM
from models.depth_estimator.depth_loss import depth_estimator
from models.vgg_extractor.vgg_depth_loss import vgg_depth_loss
from models.modules.ssim_loss import SSIM
from models.modules.mask_loss import mask_loss, dynamic_weight_mask_loss
from models.modules.replicate import patch_replication_callback
from torch.optim import lr_scheduler as lr_scheduler_torch
import numpy as np

logger = logging.getLogger('base')


class F_Model_depthSeg(BaseModel):
    def __init__(self, opt):
        super(F_Model_depthSeg, self).__init__(opt)
        # define some variable
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        self.segNet = networks.define_SegNet(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
            self.segNet = DistributedDataParallel(self.segNet, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
            self.segNet = DataParallel(self.segNet)
            patch_replication_callback(self.segNet)
        # self.netG.to(self.device)
        # print network
        self.print_network()
        self.load()

        if self.is_train: 
            train_opt = opt['train']
            #self.init_model() # Not use init is OK, since Pytorch has its owen init (by default)
            self.netG.train()
            self.segNet.train()
            
            # ResBlk num
            # self.n_depthResBlk = opt['network_G']['n_depthResBlk']

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']
            
            # depth loss
            self.use_depth_loss = train_opt['depth_loss']['use_depth_criterion']
            if self.use_depth_loss:
                self.depth_estimator = depth_estimator(train_opt['depth_loss'], device=self.device)

            # vgg loss
            self.use_vgg_loss = train_opt['vgg_loss']['use_vgg_criterion']
            if self.use_vgg_loss:
                self.vgg_depth_loss = vgg_depth_loss(train_opt['vgg_loss'], device=self.device)

            # SSIM loss
            self.use_ssim_loss = train_opt['ssim_loss']['use_ssim_criterion']
            if self.use_ssim_loss :
                self.cri_ssim = SSIM()
                self.l_ssim_w = train_opt['ssim_loss']['ssim_weight']

            # Mask loss 
            self.use_mask_loss = train_opt['mask_loss']['use_mask_criterion']
            if self.use_mask_loss :
                self.mask_loss = mask_loss(train_opt['mask_loss'])

            self.use_dynamic_loss = train_opt['dynamic_loss']['use_dynamic_criterion']
            if self.use_dynamic_loss:
                self.dynamic_loss = dynamic_weight_mask_loss(train_opt['dynamic_loss'], num_trainable_para=opt['datasets']['train']['depthMaskNum'])
            # optimizers for generator
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            
            # append trainable parameters for dynamic loss
            if self.use_dynamic_loss:
                for k, v in self.dynamic_loss.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            #self.optimizer_G = torch.optim.SGD(optim_params, lr=train_opt['lr_G'], momentum=0.9)
            self.optimizers.append(self.optimizer_G)
            # schedulers for generator
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.MultiStepLR(optimizer, train_opt['lr_steps'],
                                                         gamma=train_opt['lr_gamma']))
                    # self.schedulers.append(
                    #     lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                    #                                      restarts=train_opt['restarts'],
                    #                                      weights=train_opt['restart_weights'],
                    #                                      gamma=train_opt['lr_gamma'],
                    #                                      clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                print('MultiStepLR learning rate scheme is enough.')

            # optimizers for segmentation network
            train_opt_seg = train_opt['segNet']
            self.seg_num_classes = opt['network_SegNet']['num_classes']
            self.optimizer_segNet = torch.optim.RMSprop(self.segNet.parameters(), lr=train_opt_seg['lr'], momentum=train_opt_seg['momentum'], weight_decay=train_opt_seg['weight_decay'])

            # scheduler for segmentation network
            scheduler_segNet = lr_scheduler_torch.StepLR(self.optimizer_segNet, step_size=train_opt_seg['setp_size'], gamma=train_opt_seg['gamma']) 
            self.schedulers.append(scheduler_segNet)
            self.segLoss = nn.BCEWithLogitsLoss().to(self.device)

            self.log_dict = OrderedDict()

    def init_model(self, scale=0.1):
        # Common practise for initialization.
        for layer in self.netG.modules():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_in')
                layer.weight.data *= scale  # for residual block
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_in')
                layer.weight.data *= scale
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                init.constant_(layer.weight, 1)
                init.constant_(layer.bias.data, 0.0)


    def feed_data(self, data):#, LR_img, ker_map):
        self.var_L = data['LQ'].to(self.device)  # LQ
        self.real_H = data['GT'].to(self.device)  # GT
        self.var_depth = data['Depth'].to(self.device) # depth map
        self.var_depthMask = data['DepthMaskList'].to(self.device) 
        self.var_segLabel_onehot = data['Seg_onehot'].to(self.device)
        self.var_segLabel = data['Seg'].to(self.device)
        # self.var_depth_x4 = data['Depth_x4'].to(self.device) # depth map
        # self.var_depth_x2 = data['Depth_x2'].to(self.device) # depth map
        # self.var_L, self.ker = LR_img.to(self.device), ker_map.to(self.device)
        # self.real_ker = data['real_ker'].to(self.device)  # real kernel map
        #self.ker = data['ker'].to(self.device) # [Batch, 1, k]
        #m = self.ker.shape[0]
        #self.ker = self.ker.view(m, -1)

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L, self.var_depth,self.var_depthMask)#, self.var_depth_x4, self.var_depth_x2)#, self.ker)

        total_loss = 0
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        total_loss += l_pix

        # use_depth_criterion
        if self.use_depth_loss:
            l_depth, l_depth_list, sr_disp, hr_disp = self.depth_estimator.calculate_depth_loss(self.fake_H, self.real_H)
            total_loss += l_depth

        # use_vgg_criterion
        if self.use_vgg_loss:
            l_vgg, l_vgg_list = self.vgg_depth_loss.get_loss(self.fake_H, self.real_H)
            total_loss += l_vgg

        # use ssim loss
        if self.use_ssim_loss:
            l_ssim = self.l_ssim_w * self.cri_ssim(self.fake_H, self.real_H)
            total_loss +=l_ssim

        # use mask loss
        if self.use_mask_loss:
            l_mask = self.mask_loss.get_mask_loss(self.fake_H, self.real_H, self.var_depthMask)
            total_loss += l_mask

        # use dynamic loss
        if self.use_dynamic_loss:
            raw_loss_list, weighted_loss_list, l_dynamic, trainable_param = self.dynamic_loss(self.fake_H, self.real_H, self.var_depthMask)
            total_loss += l_dynamic
        # total_loss.backward()
        # self.optimizer_G.step()

        # update segmentation network
        self.optimizer_segNet.zero_grad()
        # new_fake = self.netG(self.var_L, self.var_depth,self.var_depthMask)
        self.pred_mask = self.segNet(self.fake_H)
        seg_loss = self.segLoss(self.pred_mask, self.var_segLabel_onehot)
        # seg_loss.backward()
        # self.optimizer_segNet.step()

        # update together
        whole_loss = total_loss + seg_loss
        whole_loss.backward()
        self.optimizer_G.step()
        self.optimizer_segNet.step()

        # soft_predicts= nn.Softmax(dim=1)(self.pred_mask)
        # dice_loss = self.Jaccard_loss_cal(self.var_segLabel.long(), soft_predicts, eps=1e-7)
        
        # set log
        ## for segmentation
        self.log_dict['l_segBCE'] = seg_loss.item()
        # self.log_dict['l_segDice'] = dice_loss

        self.log_dict['l_all'] = total_loss.item()
        self.log_dict['l_pix'] = l_pix.item()
        if self.use_depth_loss:
            self.log_dict['l_depth'] = l_depth.item()
            self.log_dict['l_depth_0'] = l_depth_list[0].item()
            self.log_dict['l_depth_1'] = l_depth_list[1].item()
            self.log_dict['l_depth_2'] = l_depth_list[2].item()
            self.log_dict['l_depth_3'] = l_depth_list[3].item()
            if step % 1000 == 0:
                # save disp
                print("Saveing the depth map for SR and HR images......")
                for i in range(4):
                    np.save('./tmp/sr_' + str(i) + '.npy', sr_disp[i].cpu().detach().numpy())
                    np.save('./tmp/hr_' + str(i) + '.npy', hr_disp[i].cpu().detach().numpy())

        if self.use_vgg_loss:
            self.log_dict['l_vgg'] = l_vgg.item()
            self.log_dict['l_vgg_0'] = l_vgg_list[0].item()
            self.log_dict['l_vgg_1'] = l_vgg_list[1].item()
            self.log_dict['l_vgg_2'] = l_vgg_list[2].item()
            self.log_dict['l_vgg_3'] = l_vgg_list[3].item()

        if self.use_ssim_loss:
            self.log_dict['l_ssim'] = l_ssim.item()

        if self.use_mask_loss:
            self.log_dict['l_mask'] = l_mask.item()
        
        if self.use_dynamic_loss:
            self.log_dict['l_dynamic'] = l_dynamic.item()
            for idx in range(len(trainable_param)):
                self.log_dict['dyn_w_'+str(idx)] = trainable_param[idx]
                self.log_dict['dyn_l_'+str(idx)] = raw_loss_list[idx]

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_SR= self.netG(self.var_L, self.var_depth, self.var_depthMask)#, self.ker)
            self.fake_Seg = self.segNet(self.fake_SR)


        self.netG.train()

    def test_x8(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.netG.eval()

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [self.var_L]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        with torch.no_grad():
            sr_list = [self.netG(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.fake_H = output_cat.mean(dim=0, keepdim=True)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_SR.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        # out_dict['ker'] = self.ker.detach()[0].float().cpu()
        out_dict['Batch_SR'] = self.fake_SR.detach().float().cpu() # Batch SR, for train
        out_dict['Depth'] = self.var_depth.detach()[0].float().cpu()
        # out_dict['Depth_x4'] = self.var_depth_x4.detach()[0].float().cpu()
        # out_dict['Depth_x2'] = self.var_depth_x2.detach()[0].float().cpu()
        out_dict['Seg'] = self.fake_Seg.detach().float().cpu()
        out_dict['Seg_label'] = self.var_segLabel.detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        # load pretrained segmentation network
        load_path_SegNet = self.opt['path']['pretrain_model_SegNet']
        if load_path_SegNet is not None:
            logger.info('Loading model for segmentation network [{:s}] ...'.format(load_path_SegNet))
            self.load_network(load_path_SegNet, self.segNet, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
        self.save_network(self.segNet, 'segNet', iter_label)

    def Jaccard_loss_cal(self, true, logits, eps=1e-7):
        """Computes the Jaccard loss, a.k.a the IoU loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the jaccard loss so we
        return the negated jaccard loss.
        Args:
            true: a tensor of shape [B, H, W] or [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            jacc_loss: the Jaccard loss.
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = logits[:,1,:,:].to(0)
        true_1_hot = true_1_hot.type(logits.type())[:,1,:,:].to(0)
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dim=(1,2))
        cardinality = torch.sum(probas + true_1_hot, dim=(1,2))
        union = cardinality - intersection
        jacc_loss = (intersection / (union + eps)).mean()
        return (1. - jacc_loss)
        