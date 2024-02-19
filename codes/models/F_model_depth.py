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


logger = logging.getLogger('base')


class F_Model_depth(BaseModel):
    def __init__(self, opt):
        super(F_Model_depth, self).__init__(opt)
        # define some variable
        self.pred_depth = opt['network_G']['predict_depth_map']
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train: 
            train_opt = opt['train']
            #self.init_model() # Not use init is OK, since Pytorch has its owen init (by default)
            self.netG.train()
            
            # ResBlk num
            self.n_depthResBlk = opt['network_G']['n_depthResBlk']

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
            if self.pred_depth:
                self.cri_depth_l1 = nn.L1Loss().to(self.device)
                self.cri_depth_ssim = SSIM().to(self.device)
                self.l_depth_l1_w = train_opt['depth_l1_weight']
                self.l_depth_ssim_w = train_opt['depth_ssim_weight']
            

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            #self.optimizer_G = torch.optim.SGD(optim_params, lr=train_opt['lr_G'], momentum=0.9)
            self.optimizers.append(self.optimizer_G)

            # schedulers
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
        self.var_depth_x8 = data['Depth_x8'].to(self.device) # depth map
        self.var_depth_x4 = data['Depth_x4'].to(self.device) # depth map
        self.var_depth_x2 = data['Depth_x2'].to(self.device) # depth map
        # self.var_L, self.ker = LR_img.to(self.device), ker_map.to(self.device)
        # self.real_ker = data['real_ker'].to(self.device)  # real kernel map
        #self.ker = data['ker'].to(self.device) # [Batch, 1, k]
        #m = self.ker.shape[0]
        #self.ker = self.ker.view(m, -1)

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        l_total = 0
        if self.pred_depth: 
            self.fake_H, self.depth_x4, self.depth_x2 = self.netG(self.var_L, self.var_depth_x8)#, self.var_depth_x4, self.var_depth_x2)#, self.ker)
            l_depth_l1 = 0
            l_depth_ssim = 0
            l_depth = 0
            if self.n_depthResBlk >= 2:
                l1_x4 = self.cri_depth_l1(self.depth_x4, self.var_depth_x4)
                ssim_x4 = self.cri_depth_ssim(self.depth_x4, self.var_depth_x4).mean()#1, True)
                l_depth_l1 += l1_x4 * self.l_depth_l1_w
                l_depth_ssim += ssim_x4 * self.l_depth_ssim_w
            if self.n_depthResBlk == 3:
                l1_x2 = self.cri_depth_l1(self.depth_x2, self.var_depth_x2)
                ssim_x2 = self.cri_depth_ssim(self.depth_x2, self.var_depth_x2).mean()#1, True)
                l_depth_l1 += l1_x2 * self.l_depth_l1_w
                l_depth_ssim += ssim_x2 * self.l_depth_ssim_w
            l_depth = l_depth_l1 + l_depth_ssim
            l_total += l_depth
        else:  
            self.fake_H = self.netG(self.var_L, self.var_depth_x8)#, self.var_depth_x4, self.var_depth_x2)#, self.ker)

        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_total += l_pix
        l_total.backward()
        self.optimizer_G.step()
        # set log
        self.log_dict['l_total'] = l_total
        self.log_dict['l_pix'] = l_pix.item()
        if self.pred_depth:
            self.log_dict['l_depth_l1'] = l_depth_l1#.item()
            self.log_dict['l_depth_ssim'] = l_depth_ssim#.item()
            self.log_dict['l_depth'] = l_depth#.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            if self.pred_depth:
                self.fake_SR, self.pred_depth_x4, self.pred_depth_x2 = self.netG(self.var_L, self.var_depth_x8)#, self.var_depth_x4, self.var_depth_x2)#, self.ker)
            else:
                self.fake_SR= self.netG(self.var_L, self.var_depth_x8)#, self.ker)



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
        out_dict['Depth_x8'] = self.var_depth_x8.detach()[0].float().cpu()
        out_dict['Depth_x4'] = self.var_depth_x4.detach()[0].float().cpu()
        out_dict['Depth_x2'] = self.var_depth_x2.detach()[0].float().cpu()
        # predicted depth map
        if self.pred_depth:
            if self.n_depthResBlk >=2 :
                out_dict['pred_Depth_x4'] = self.pred_depth_x4.detach()[0].float().cpu()
            if self.n_depthResBlk == 3:
                out_dict['pred_Depth_x2'] = self.pred_depth_x2.detach()[0].float().cpu()
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

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
