import torch
import logging
import models.modules.SRResNet_arch as SRResNet_arch
import models.modules.discriminator_vgg_arch as SRGAN_arch
import models.modules.RRDBNet_arch as RRDBNet_arch
import models.modules.sftmd_arch as sftmd_arch
from models.modules.fcn import FCN8s
logger = logging.getLogger('base')


####################
# define network
####################
#### Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'MSRResNet':
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'RRDBNet':
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])
    elif which_model == 'Predictor':
        netG = sftmd_arch.Predictor(in_nc=opt_net['in_nc'], nf=opt_net['nf'], code_len=opt_net['code_length'])
    elif which_model == 'Corrector':
        netG = sftmd_arch.Corrector(in_nc=opt_net['in_nc'], nf=opt_net['nf'], code_len=opt_net['code_length'])
    elif which_model == 'SFTMD':
        netG = sftmd_arch.SFTMD_noKernel(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                nf=opt_net['nf'], nb=opt_net['nb'], scale=opt_net['upscale'], input_para=opt_net['code_length'])
    elif which_model == 'SFTMD_upsacle_after_ResBlk':
        netG = sftmd_arch.SFTMD_upsacle_after_ResBlk(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                nf=opt_net['nf'], nb=opt_net['nb'], scale=opt_net['upscale'], input_para=opt_net['code_length'])
    elif which_model == 'SFTMD_upsacle_after_ResBlk_depth':
        netG = sftmd_arch.SFTMD_upsacle_after_ResBlk_depth(pred_depth=opt_net['predict_depth_map'], n_depthResBlk=opt_net['n_depthResBlk'], use_attention=opt_net['use_attention'], in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                nf=opt_net['nf'], nb=opt_net['nb'], scale=opt_net['upscale'], input_para=opt_net['code_length'])
    elif which_model == 'SFTMD_upsacle_after_ResBlk_depth_condition':
        netG = sftmd_arch.SFTMD_upsacle_after_ResBlk_depth_condition(which_ResBlk_depth=opt_net['which_ResBlk_depth'], in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                nf=opt_net['nf'], nb=opt_net['nb'], scale=opt_net['upscale'], input_para=opt_net['code_length'])
    elif which_model == 'DepthNet':
        datalist = list(opt['datasets'].items())
        print(datalist[0])
        if datalist[0][0] == 'train':
            depthRangeNum = opt['datasets']['train']['depthMaskNum']
        else:
            depthRangeNum = opt['datasets']['test_1']['depthMaskNum']
        netG = sftmd_arch.DepthNet(which_ResBlk_depth=opt_net['which_ResBlk_depth'], in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                nf=opt_net['nf'], nb=opt_net['nb'], scale=opt_net['upscale'], input_para=opt_net['code_length'], depth_latent_ch=opt_net['depth_latent_ch'], depthRangeNum=depthRangeNum,norm_type=opt_net['norm_type'], use_trainable_params=opt_net['use_trainable_params'], norm_gamma=opt_net['norm_gamma'], norm_beta=opt_net['norm_beta'],ablate_depth_block=opt_net['ablate_depth_block'],ablate_depth_matrix=opt_net['ablate_depth_matrix'])
    elif which_model == 'SRResNet':
        netG = sftmd_arch.SRResNet()
    elif which_model == 'SFTMD_DEMO':
        netG = sftmd_arch.SFTMD_DEMO(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                nf=opt_net['nf'], nb=opt_net['nb'], scale=opt_net['upscale'], input_para=opt_net['code_length'])
    # elif which_model == 'sft_arch':  # SFT-GAN
    #     netG = sft_arch.SFT_Net()
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG


#### Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


#### Define Network used for Perceptual Loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF

#### Define Segmentation Network 
def define_SegNet(opt):
    opt_net = opt['network_SegNet']
    # which_model = opt_net['which_model_SegNet']
    netSeg = FCN8s(opt_net['num_classes'])
    return netSeg
