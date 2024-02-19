'''
architecture for sftmd
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.normalization import SEAN
import math

class Predictor(nn.Module):
    def __init__(self, in_nc=3, nf=64, code_len=10, use_bias=True):
        super(Predictor, self).__init__()

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(in_nc, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, code_len, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ])
        #   self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.globalPooling = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, input):
        conv = self.ConvNet(input)
        flat = self.globalPooling(conv)
        return flat.view(flat.size()[:2]) # torch size: [B, code_len]



class Corrector(nn.Module):
    def __init__(self, in_nc=3, nf=64, code_len=10, use_bias=True):
        super(Corrector, self).__init__()

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(in_nc, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ])

        self.code_dense = nn.Sequential(*[
            nn.Linear(code_len, nf, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Linear(nf, nf, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ])

        self.global_dense = nn.Sequential(*[
            nn.Conv2d(nf * 2, nf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, code_len, kernel_size=1, stride=1, padding=0, bias=use_bias),
        ])

        self.nf = nf
        self.globalPooling = nn.AdaptiveAvgPool2d([1, 1])

    def forward(self, input, code, res=False):
        conv_input = self.ConvNet(input)
        B, C_f, H_f, W_f = conv_input.size() # LR_size

        conv_code = self.code_dense(code).view((B, self.nf, 1, 1)).expand((B, self.nf, H_f, W_f)) # h_stretch
        conv_mid = torch.cat((conv_input, conv_code), dim=1)
        code_res = self.global_dense(conv_mid)

        # Delta_h_p
        flat = self.globalPooling(code_res)
        Delta_h_p = flat.view(flat.size()[:2])

        if res:
            return Delta_h_p
        else:
            return Delta_h_p + code


class SFT_Layer(nn.Module):
    def __init__(self, nf=64, para=10):
        super(SFT_Layer, self).__init__()
        self.mul_conv1 = nn.Conv2d(para + nf, 32, kernel_size=3, stride=1, padding=1)
        self.mul_leaky = nn.LeakyReLU(0.2)
        self.mul_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

        self.add_conv1 = nn.Conv2d(para + nf, 32, kernel_size=3, stride=1, padding=1)
        self.add_leaky = nn.LeakyReLU(0.2)
        self.add_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

    def forward(self, feature_maps, para_maps):
        cat_input = torch.cat((feature_maps, para_maps), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.mul_leaky(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.add_leaky(self.add_conv1(cat_input)))
        return feature_maps * mul + add


class SFT_Residual_Block(nn.Module):
    def __init__(self, nf=64, para=10):
        super(SFT_Residual_Block, self).__init__()
        self.sft1 = SFT_Layer(nf=nf, para=para)
        self.sft2 = SFT_Layer(nf=nf, para=para)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, feature_maps, para_maps):
        fea1 = F.relu(self.sft1(feature_maps, para_maps))
        fea2 = F.relu(self.sft2(self.conv1(fea1), para_maps))
        fea3 = self.conv2(fea2)
        return torch.add(feature_maps, fea3)

class Classic_Residual_Block(nn.Module):
    def __init__(self, wn, nf=64, norm_type='weight_norm'):
        super(Classic_Residual_Block, self).__init__()
        if norm_type == 'weight_norm':
            block = [
                wn(nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True)),
                nn.ReLU(True),
                wn(nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True))
            ]
        else:
            block = [
                nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True),
                nn.InstanceNorm2d(nf, affine=True, track_running_stats=True),
                nn.ReLU(True),
                nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True),
                nn.InstanceNorm2d(nf, affine=True, track_running_stats=True)
            ]

        self.block = nn.Sequential(*block)
    def forward(self, feature_maps):
        fea = self.block(feature_maps)
        out = torch.add(feature_maps, fea)
        out = F.relu(out)
        return out

class PositionAttentionModule(nn.Module):
    """ Position attention module"""
    def __init__(self, in_channels, **kwargs):
        super(PositionAttentionModule, self).__init__()
        self.conv_a = nn.Sequential(
            nn.Conv2d(1, in_channels, 1),
            nn.ReLU()
        )
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature_maps, depth):
        depth_feat = self.conv_a(depth)

        batch_size, _, height, width = feature_maps.size()

        feat_b = self.conv_b(feature_maps).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(depth_feat).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(depth_feat).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)

        return feat_e

class PositionAttentionModule_efficient(nn.Module):
    """ Position attention module"""
    def __init__(self, in_channels, **kwargs):
        super(PositionAttentionModule_efficient, self).__init__()
        self.conv_a = nn.Sequential(
            nn.Conv2d(1, in_channels, 1),
            nn.ReLU()
        )
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature_maps, depth):
        depth_feat = self.conv_a(depth)

        batch_size, _, height, width = feature_maps.size()

        feat_b = self.conv_b(feature_maps).view(batch_size, -1, height * width).permute(0, 2, 1) # bsx(hxw)xch/8
        feat_c = self.conv_c(depth_feat).view(batch_size, -1, height * width) # bsxch/8x(hxw)
        feat_d = self.conv_d(depth_feat).view(batch_size, -1, height * width) # bsxchx(hxw)

        attention_s = self.softmax(torch.bmm(feat_d, feat_b))
        feat_e = torch.bmm(attention_s, feat_c).view(batch_size, -1, height, width)

        return feat_e
class SPADE(nn.Module):
    def __init__(self, nf, in_channels=1, ks=3, pw=1, param_free_norm_type='instance', use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        if self.use_attention:
            self.attenModule = PositionAttentionModule_efficient(nf)

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(nf, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(nf, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(nf, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = nf
        # ks = 3
        # pw = 1 #ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(in_channels, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, nf, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, nf, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        # upsampling the depth map
        if segmap.shape[2] != x.shape[2]:
            segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')

        # use the attention module before normalization
        if self.use_attention:
            x = self.attenModule(x, segmap)

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class Depth_Residual_Block(nn.Module):
    def __init__(self, depth_ch=1, nf=64, use_attention=False):
        super(Depth_Residual_Block, self).__init__()
        conv1 = [
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(nf, affine=False)
        ]
        self.norm1 = SPADE(nf, in_channels=depth_ch, use_attention=use_attention)
        self.actv1 = nn.ReLU(True)

        conv2 = [
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(nf, affine=False)]
        self.norm2 = SPADE(nf, in_channels=depth_ch, use_attention=use_attention)    

        self.conv1 = nn.Sequential(*conv1)
        self.conv2 = nn.Sequential(*conv2)

    def forward(self, feature_maps, depth_map):
        fea_norm = self.norm1(self.conv1(feature_maps), depth_map)
        fea_norm_actv = self.actv1(fea_norm)

        fea_norm2 = self.norm2(self.conv2(fea_norm_actv), depth_map)

        out = torch.add(feature_maps, fea_norm2)
        out = F.relu(out)
        return out


class SFTMD_upsacle_after_ResBlk_depth_condition(nn.Module):
    def __init__(self, which_ResBlk_depth=[], in_nc=3, out_nc=3, nf=64, nb=16, scale=4, input_para=10, min=0.0, max=1.0):
        super(SFTMD_upsacle_after_ResBlk_depth_condition, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.num_blocks = nb
        self.which_ResBlk_depth = which_ResBlk_depth
        # weight normalization
        wn = lambda x: torch.nn.utils.weight_norm(x)

        head = [
            wn(nn.Conv2d(in_nc, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(0.2),
            wn(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(0.2),
            wn(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(0.2)
        ]
        self.head = nn.Sequential(*head)

        depth_ch = 64
        self.depth_condition = nn.Sequential(
            wn(nn.Conv2d(1, depth_ch, 3, stride=1, padding=1)),
            nn.LeakyReLU(0.2),
            wn(nn.Conv2d(depth_ch, depth_ch, 3, stride=1, padding=1)),
            nn.LeakyReLU(0.2),
            wn(nn.Conv2d(depth_ch, depth_ch, 3, stride=1, padding=1)),
            nn.LeakyReLU(0.2)
        )

        for i in range(nb):
            input_ch = nf
            if i > nb-3:
                input_ch = 32
            if i in self.which_ResBlk_depth:
                self.add_module('depth-residual' + str(i + 1), Depth_Residual_Block(nf=input_ch,depth_ch=depth_ch))
            else:
                self.add_module('classic-residual' + str(i + 1), Classic_Residual_Block(wn,nf=input_ch))

        upscale1 = [
            wn(nn.Conv2d(in_channels=64, out_channels=64 * 4, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            wn(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True)
            ]
        upscale2 = [
            wn(nn.Conv2d(in_channels=32, out_channels=32 * 4, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            wn(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True)]
        upscale3 = [
            wn(nn.Conv2d(in_channels=32, out_channels=32 * 4, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)]
        self.upscale1 = nn.Sequential(*upscale1)
        self.upscale2 = nn.Sequential(*upscale2)
        self.upscale3 = nn.Sequential(*upscale3)
        
        self.conv_output = nn.Conv2d(in_channels=32, out_channels=out_nc, kernel_size=9, stride=1, padding=4, bias=True)
        
    def forward(self, input, depth):
        # B, C, H, W = input.size() 

        fea_bef = self.head(input)
        fea_in = fea_bef
        # feature of depth map
        depth_feat = self.depth_condition(depth)   

        for i in range(self.num_blocks-3):
            if i in self.which_ResBlk_depth:
                fea_in = self.__getattr__('depth-residual' + str(i+1))(fea_in, depth_feat)
            else:
                fea_in = self.__getattr__('classic-residual' + str(i + 1))(fea_in)
        
        fea_mid = fea_in

        feat_add1 = torch.add(fea_mid, fea_bef)
        feat_up1 = self.upscale1(feat_add1)

        if self.num_blocks-2 in self.which_ResBlk_depth:
            feat_up1 = self.__getattr__('depth-residual' + str(self.num_blocks - 1))(feat_up1, depth_feat)
        else:
            feat_up1 = self.__getattr__('classic-residual' + str(self.num_blocks - 1))(feat_up1)
        
        feat_up2 = self.upscale2(feat_up1)
        if self.num_blocks-1 in self.which_ResBlk_depth:
            feat_up2 = self.__getattr__('depth-residual' + str(self.num_blocks))(feat_up2, depth_feat)
        else:
            feat_up2 = self.__getattr__('classic-residual' + str(self.num_blocks))(feat_up2)

        feat_up3 = self.upscale3(feat_up2)
        
        out = self.conv_output(feat_up3)

        return torch.clamp(out, min=self.min, max=self.max)


class SFTMD_upsacle_after_ResBlk_depth(nn.Module):
    def __init__(self, pred_depth=False, n_depthResBlk=3, use_attention=False, in_nc=3, out_nc=3, nf=64, nb=16, scale=4, input_para=10, min=0.0, max=1.0):
        super(SFTMD_upsacle_after_ResBlk_depth, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.num_blocks = nb
        self.n_depthResBlk = n_depthResBlk
        self.pred_depth = pred_depth
        # weight normalization
        wn = lambda x: torch.nn.utils.weight_norm(x)

        head = [
            wn(nn.Conv2d(in_nc, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(0.2),
            wn(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(0.2),
            wn(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(0.2)
        ]
        self.head = nn.Sequential(*head)

        for i in range(nb-4):
            self.add_module('classic-residual' + str(i + 1), Classic_Residual_Block(wn,nf=nf))

        if self.n_depthResBlk >= 1:
            self.add_module('depth-residual' + str(nb-3), Depth_Residual_Block(nf=nf,use_attention=use_attention))
        else:
            self.add_module('classic-residual' + str(nb-3), Classic_Residual_Block(wn,nf=nf))

        if self.n_depthResBlk >= 2:
            self.add_module('depth-residual' + str(nb-2), Depth_Residual_Block(nf=32,use_attention=use_attention))
        else:
            self.add_module('classic-residual' + str(nb-2), Classic_Residual_Block(wn,nf=32))

        if self.n_depthResBlk >= 3:
            self.add_module('depth-residual' + str(nb-1), Depth_Residual_Block(nf=32,use_attention=use_attention))
        else:
            self.add_module('classic-residual' + str(nb-1), Classic_Residual_Block(wn,nf=32))

        upscale1 = [
            wn(nn.Conv2d(in_channels=64, out_channels=64 * 4, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            wn(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True)
            ]
        upscale2 = [
            wn(nn.Conv2d(in_channels=32, out_channels=32 * 4, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            wn(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True)]
        upscale3 = [
            wn(nn.Conv2d(in_channels=32, out_channels=32 * 4, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)]
        self.upscale1 = nn.Sequential(*upscale1)
        self.upscale2 = nn.Sequential(*upscale2)
        self.upscale3 = nn.Sequential(*upscale3)
        
        self.conv_output = nn.Conv2d(in_channels=32, out_channels=out_nc, kernel_size=9, stride=1, padding=4, bias=True)

        if self.pred_depth:
            # upscale the depth map
            # N x 1 x 128 x 128 - > N x 1 x 256 x 256
            nf_depth = 64
            self.depth_upscale1 = nn.Sequential(
                wn(nn.Conv2d(1, nf_depth, 3, stride=1, padding=1)),
                nn.LeakyReLU(0.2),
                Classic_Residual_Block(wn,nf=nf_depth),
                wn(nn.Conv2d(in_channels=nf_depth, out_channels=nf_depth * 4, kernel_size=3, stride=1, padding=1, bias=True)),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=nf_depth, out_channels=1, kernel_size=9, stride=1, padding=4, bias=True),
                nn.Sigmoid()
            )

            # N x 1 x 256 x 256 -> N x 1 x 512 x 512
            nf_depth = 64
            self.depth_upscale2 = nn.Sequential(
                wn(nn.Conv2d(1, nf_depth, 3, stride=1, padding=1)),
                nn.LeakyReLU(0.2),
                Classic_Residual_Block(wn,nf=nf_depth),
                wn(nn.Conv2d(in_channels=nf_depth, out_channels=nf_depth * 4, kernel_size=3, stride=1, padding=1, bias=True)),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=nf_depth, out_channels=1, kernel_size=9, stride=1, padding=4, bias=True),
                nn.Sigmoid()
            )

    def forward(self, input, depth):#, ker_code):
        B, C, H, W = input.size() 

        fea_bef = self.head(input)
        fea_in = fea_bef
        for i in range(self.num_blocks-4):
            fea_in = self.__getattr__('classic-residual' + str(i + 1))(fea_in)

        if self.n_depthResBlk >= 1:
            fea_in = self.__getattr__('depth-residual' + str(self.num_blocks - 3))(fea_in, depth)
        else:
            fea_in = self.__getattr__('classic-residual' + str(self.num_blocks))(fea_in)

        fea_mid = fea_in

        feat_add1 = torch.add(fea_mid, fea_bef)
        feat_up1 = self.upscale1(feat_add1)
        depth_x4 = None
        depth_x2 = None
        if self.n_depthResBlk >= 2:
            depth_x4 = self.depth_upscale1(depth) if self.pred_depth else depth
            feat_up1 = self.__getattr__('depth-residual' + str(self.num_blocks - 2))(feat_up1, depth_x4)
        else:
            feat_up1 = self.__getattr__('classic-residual' + str(self.num_blocks - 2))(feat_up1)
        
        feat_up2 = self.upscale2(feat_up1)
        if self.n_depthResBlk >= 3:
            depth_x2 = self.depth_upscale1(depth_x4) if self.pred_depth else depth
            feat_up2 = self.__getattr__('depth-residual' + str(self.num_blocks - 1))(feat_up2, depth_x2)
        else:
            feat_up2 = self.__getattr__('classic-residual' + str(self.num_blocks - 1))(feat_up2)

        feat_up3 = self.upscale3(feat_up2)
        
        
        # fea = self.conv_mid(feat_up3)
        out = self.conv_output(feat_up3)

        if self.pred_depth:
            return torch.clamp(out, min=self.min, max=self.max), depth_x4, depth_x2

        return torch.clamp(out, min=self.min, max=self.max)


class SFTMD_upsacle_after_ResBlk(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, scale=4, input_para=10, min=0.0, max=1.0):
        super(SFTMD_upsacle_after_ResBlk, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.num_blocks = nb
        # weight normalization
        wn = lambda x: torch.nn.utils.weight_norm(x)

        head = [
            wn(nn.Conv2d(in_nc, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(0.2),
            wn(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(0.2),
            wn(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(0.2)
        ]
        self.head = nn.Sequential(*head)

        for i in range(nb-3):
            self.add_module('classic-residual' + str(i + 1), Classic_Residual_Block(wn,nf=nf))
        
        self.add_module('classic-residual' + str(nb-2), Classic_Residual_Block(wn,nf=32))
        self.add_module('classic-residual' + str(nb-1), Classic_Residual_Block(wn,nf=32))

        # self.conv_mid = nn.Sequential(
        #     wn(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)),
        #     nn.LeakyReLU(0.2)
        # )
        #scale == 8
        upscale1 = [
            wn(nn.Conv2d(in_channels=64, out_channels=64 * 4, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            wn(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True)
            ]
        upscale2 = [
            wn(nn.Conv2d(in_channels=32, out_channels=32 * 4, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            wn(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True)]
        upscale3 = [
            wn(nn.Conv2d(in_channels=32, out_channels=32 * 4, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)]
        self.upscale1 = nn.Sequential(*upscale1)
        self.upscale2 = nn.Sequential(*upscale2)
        self.upscale3 = nn.Sequential(*upscale3)
        
        # elif scale == 4: #x4
        #     print("scale == 4")
        #     upscale = [
        #         nn.Conv2d(in_channels=64, out_channels=64 * scale, kernel_size=3, stride=1, padding=1, bias=True),
        #         nn.PixelShuffle(scale // 2),
        #         nn.LeakyReLU(0.2, inplace=True),
        #         nn.Conv2d(in_channels=64, out_channels=64 * scale, kernel_size=3, stride=1, padding=1, bias=True),
        #         nn.PixelShuffle(scale // 2),
        #         nn.LeakyReLU(0.2, inplace=True)
        #     ]
        # else: #x2, x3
        #     upscale = [
        #         nn.Conv2d(in_channels=64, out_channels=64*scale**2, kernel_size=3, stride=1, padding=1, bias=True),
        #         nn.PixelShuffle(scale),
        #         nn.LeakyReLU(0.2, inplace=True)
        #     ]

        self.conv_output = nn.Conv2d(in_channels=32, out_channels=out_nc, kernel_size=9, stride=1, padding=4, bias=True)

    def forward(self, input):#, ker_code):
        B, C, H, W = input.size() 

        fea_bef = self.head(input)
        fea_in = fea_bef
        for i in range(self.num_blocks-3):
            fea_in = self.__getattr__('classic-residual' + str(i + 1))(fea_in)
        fea_mid = fea_in

        feat_add1 = torch.add(fea_mid, fea_bef)
        feat_up1 = self.upscale1(feat_add1)
        feat_up1 = self.__getattr__('classic-residual' + str(self.num_blocks - 2))(feat_up1)
        
        feat_up2 = self.upscale2(feat_up1)
        feat_up2 = self.__getattr__('classic-residual' + str(self.num_blocks - 1))(feat_up2)

        feat_up3 = self.upscale3(feat_up2)
        
        
        # fea = self.conv_mid(feat_up3)
        out = self.conv_output(feat_up3)

        return torch.clamp(out, min=self.min, max=self.max)

class SFTMD_noKernel(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, scale=4, input_para=10, min=0.0, max=1.0):
        super(SFTMD_noKernel, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.num_blocks = nb
        # weight normalization
        wn = lambda x: torch.nn.utils.weight_norm(x)

        head = [
            wn(nn.Conv2d(in_nc, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(0.2),
            wn(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(0.2),
            wn(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(0.2)
        ]
        self.head = nn.Sequential(*head)


        # self.conv1 = nn.Conv2d(in_nc, 64, 3, stride=1, padding=1)
        # self.relu_conv1 = nn.LeakyReLU(0.2)
        # self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        # self.relu_conv2 = nn.LeakyReLU(0.2)
        # self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        
        #sft_branch is not used.
        # sft_branch = []
        # for i in range(nb):
        #     sft_branch.append(SFT_Residual_Block())
        # self.sft_branch = nn.Sequential(*sft_branch)


        for i in range(nb):
            self.add_module('classic-residual' + str(i + 1), Classic_Residual_Block(wn,nf=nf))

        # self.sft = SFT_Layer(nf=64, para=input_para)
        # self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_mid = nn.Sequential(
            wn(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(0.2)
        )
        if scale == 8:
            upscale = [
                wn(nn.Conv2d(in_channels=64, out_channels=64 * 4, kernel_size=3, stride=1, padding=1, bias=True)),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                wn(nn.Conv2d(in_channels=64, out_channels=64 * 4, kernel_size=3, stride=1, padding=1, bias=True)),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                wn(nn.Conv2d(in_channels=64, out_channels=64 * 4, kernel_size=3, stride=1, padding=1, bias=True)),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        elif scale == 4: #x4
            print("scale == 4")
            upscale = [
                nn.Conv2d(in_channels=64, out_channels=64 * scale, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64 * scale, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        else: #x2, x3
            upscale = [
                nn.Conv2d(in_channels=64, out_channels=64*scale**2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        self.upscale = nn.Sequential(*upscale)
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=out_nc, kernel_size=9, stride=1, padding=4, bias=True)

    def forward(self, input):#, ker_code):
        B, C, H, W = input.size() # I_LR batch
        # B_h, C_h = ker_code.size() # Batch, Len=10
        # ker_code_exp = ker_code.view((B_h, C_h, 1, 1)).expand((B_h, C_h, H, W)) #kernel_map stretch

        fea_bef = self.head(input)
        fea_in = fea_bef
        for i in range(self.num_blocks):
            fea_in = self.__getattr__('classic-residual' + str(i + 1))(fea_in)
        fea_mid = fea_in
        
        fea_add = torch.add(fea_mid, fea_bef)
        fea = self.upscale(self.conv_mid(fea_add))
        out = self.conv_output(fea)

        return torch.clamp(out, min=self.min, max=self.max)



class RegionWiseAvgPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feature_map, mask):
        if mask.size(2) != feature_map.size(2) or mask.size(3) != feature_map.size(3):
            mask = F.interpolate(mask, size=(feature_map.size(2), feature_map.size(3)), mode='bilinear',
                                 align_corners=True)
            mask = (mask >= 0.5).type_as(mask)
        out = list()
        for i in range(mask.size(1)):
            region_mask = torch.cat([mask[:, i, :, :].unsqueeze(1)] * feature_map.size(1), dim=1)
            # new version
            # print("region_mask:",region_mask.shape)
            # print("feature_map:",feature_map.shape)
            matrix = region_mask * feature_map
            # print("matrix:",matrix.shape)
            sum_feat = torch.sum(matrix, dim=(2,3))
            sum_mask = torch.sum(region_mask, dim=(2,3))
            val = sum_feat / (sum_mask+1e-10)
            out.append(val.unsqueeze(1))
            # original version
            # out.append(self.avg_pool(region_mask * feature_map).squeeze(2).squeeze(2).unsqueeze(1))
        return torch.cat(out, dim=1)

class Encoder(nn.Module):
    def __init__(self, in_nc=3, latent_ch = 256, norm_type='weight_norm',isBaseline=None):
        super(Encoder, self).__init__()
        self.isBaseline = isBaseline
        # weight normalization
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.actvn = nn.LeakyReLU(0.2, False)
        if norm_type == 'weight_norm':
            self.layer1 = wn(nn.Conv2d(in_nc, 32, 3, stride=1, padding=1))
            # downsample
            self.layer2 = wn(nn.Conv2d(32, 64, 3, stride=2, padding=1))
            self.layer3 = wn(nn.Conv2d(64, 128, 3, stride=2, padding=1))
            # upsample
            self.layer4 = wn(nn.ConvTranspose2d(128, latent_ch, 3, stride=2, padding=1))
            self.layer5 = wn(nn.Conv2d(latent_ch, latent_ch, 3, stride=2, padding=1))
        else: # instance normalization
            layer1 = [ nn.Conv2d(in_nc, 32, 3, stride=1, padding=1),
                       nn.InstanceNorm2d(32, affine=True, track_running_stats=True)]
            # downsample
            layer2 = [ nn.Conv2d(32, 64, 3, stride=2, padding=1),
                       nn.InstanceNorm2d(64, affine=True, track_running_stats=True)]
            layer3 = [ nn.Conv2d(64, 128, 3, stride=2, padding=1),
                       nn.InstanceNorm2d(128, affine=True, track_running_stats=True)]
            # upsample
            layer4 = [ nn.ConvTranspose2d(128, latent_ch, 3, stride=2, padding=1),
                       nn.InstanceNorm2d(latent_ch, affine=True, track_running_stats=True)]
            layer5 = [ nn.Conv2d(latent_ch, latent_ch, 3, stride=2, padding=1),
                       nn.InstanceNorm2d(latent_ch, affine=True, track_running_stats=True)]
            self.layer1 = nn.Sequential(*layer1)
            self.layer2 = nn.Sequential(*layer2)
            self.layer3 = nn.Sequential(*layer3)
            self.layer4 = nn.Sequential(*layer4)
            self.layer5 = nn.Sequential(*layer5)

        # Depth-wise average pooling
        self.pool = RegionWiseAvgPooling()
    def forward(self, input, depthMask):
        out = self.layer1(input)
        feat_downscaled = out
        if self.isBaseline :
            return self.actvn(feat_downscaled), None
        out = self.layer2(self.actvn(out))
        out = self.layer3(self.actvn(out))
        out = self.layer4(self.actvn(out))
        out = self.layer5(self.actvn(out))
        # pooling
        out = self.pool(feature_map=out, mask=depthMask)
        # return feat_downscaled, out
        return self.actvn(feat_downscaled), out

class Encoder_noDepthMatrix(nn.Module):
    def __init__(self, in_nc=3, latent_ch = 256, norm_type='weight_norm'):
        super(Encoder_noDepthMatrix, self).__init__()
        # weight normalization
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.actvn = nn.LeakyReLU(0.2, False)
        self.layer1 = wn(nn.Conv2d(in_nc, 32, 3, stride=1, padding=1))
        # downsample
        self.layer2 = wn(nn.Conv2d(32, 64, 3, stride=1, padding=1))
        self.layer3 = wn(nn.Conv2d(64, 128, 3, stride=2, padding=1))
        # upsample
        self.layer4 = wn(nn.ConvTranspose2d(128, 256, 3, stride=2, padding=1))
        self.layer5 = wn(nn.Conv2d(256, latent_ch, 3, stride=1, padding=1))

    def forward(self, input):
        out = self.layer1(input)
        feat_downscaled = out
        out = self.layer2(self.actvn(out))
        out = self.layer3(self.actvn(out))
        out = self.layer4(self.actvn(out))
        out = self.layer5(self.actvn(out)) # 128x128x256
        return feat_downscaled, out

class Depth_Residual_Block_Mask(nn.Module):
    def __init__(self, nf=64, depth_latent_ch=256, depthRangeNum=10, use_trainable_params=True, norm_gamma=0.1, norm_beta=0.1,ablate_depth_matrix=False,ablate_depth_block=False):
        super(Depth_Residual_Block_Mask, self).__init__()
        conv1 = [
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(nf, affine=False)
        ]
        self.norm1 = SEAN(label_nc=depthRangeNum, norm_nc=nf, len_latent=depth_latent_ch, use_trainable_params=use_trainable_params, norm_gamma=norm_gamma, norm_beta=norm_beta, ablate_depth_matrix=ablate_depth_matrix,ablate_depth_block=ablate_depth_block)
        self.actv1 = nn.ReLU(True)

        conv2 = [
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(nf, affine=False)]
        self.norm2 = SEAN(label_nc=depthRangeNum, norm_nc=nf, len_latent=depth_latent_ch, use_trainable_params=use_trainable_params, norm_gamma=norm_gamma, norm_beta=norm_beta, ablate_depth_matrix=ablate_depth_matrix,ablate_depth_block=ablate_depth_block)    

        self.conv1 = nn.Sequential(*conv1)
        self.conv2 = nn.Sequential(*conv2)

    def forward(self, feature_maps, depth_map, depthMask, depthVec):
        fea_norm = self.norm1(self.conv1(feature_maps), depth_map, depthMask, depthVec)
        fea_norm_actv = self.actv1(fea_norm)

        fea_norm2 = self.norm2(self.conv2(fea_norm_actv), depth_map, depthMask, depthVec)

        out = torch.add(feature_maps, fea_norm2)
        out = F.relu(out)
        return out


class DepthNet(nn.Module):
    def __init__(self, which_ResBlk_depth=[], in_nc=3, out_nc=3, nf=64, nb=16, scale=4, input_para=10, min=0.0, max=1.0, depth_latent_ch=256, depthRangeNum=10, norm_type='weight_norm', use_trainable_params=True, norm_gamma=0.1, norm_beta=0.1, ablate_depth_matrix=False, ablate_depth_block=False):
        
        super(DepthNet, self).__init__()
        self.scale = scale
        self.min = min
        self.max = max
        self.para = input_para
        self.num_blocks = nb
        self.which_ResBlk_depth = which_ResBlk_depth
        self.isBaseline = len(which_ResBlk_depth)==0
        self.ablate_depth_matrix = ablate_depth_matrix
        self.ablate_depth_block = ablate_depth_block
        # weight normalization
        wn = lambda x: torch.nn.utils.weight_norm(x)

        ### encoder
        if self.ablate_depth_matrix:
            self.encoder = Encoder_noDepthMatrix(in_nc=in_nc, latent_ch=depth_latent_ch,norm_type=norm_type)
        else:
            self.encoder = Encoder(in_nc=in_nc, latent_ch=depth_latent_ch,norm_type=norm_type,isBaseline=self.isBaseline)

        ### HEAD
        if norm_type == 'weight_norm':
            head = [
                wn(nn.Conv2d(32, 64, 3, stride=1, padding=1)),
                nn.LeakyReLU(0.2),
                wn(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
                nn.LeakyReLU(0.2)
            ]
        else: # Instance Normalization
            head = [
                nn.Conv2d(32, 64, 3, stride=1, padding=1),
                nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.2)
            ]

        self.head = nn.Sequential(*head)

        num_last_block = 1 if self.scale == 3 else int(math.log(self.scale, 2))
        ch_last2_upscale = 64 if self.scale == 4 else 32
        ch_last_upscale = 64 if self.scale < 4 else 32
        for i in range(nb):
            input_ch = nf
            if i > nb-num_last_block:
                input_ch = 32
            if i in self.which_ResBlk_depth:
                self.add_module('depth-residual' + str(i + 1), Depth_Residual_Block_Mask(nf=input_ch,depth_latent_ch=depth_latent_ch, depthRangeNum=depthRangeNum, use_trainable_params=use_trainable_params, norm_gamma=norm_gamma, norm_beta=norm_beta, ablate_depth_matrix=ablate_depth_matrix,ablate_depth_block=ablate_depth_block))
            else:
                self.add_module('classic-residual' + str(i + 1), Classic_Residual_Block(wn,nf=input_ch,norm_type=norm_type))
        
        self.upscale1 = nn.Sequential(
            wn(nn.Conv2d(in_channels=64, out_channels=64 * 4, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            wn(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.upscale2 = nn.Sequential(
            wn(nn.Conv2d(in_channels=ch_last2_upscale, out_channels=32 * 4, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            wn(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True))
        final_scale = 3 if self.scale == 3 else 2
        self.upscale3 = nn.Sequential(
            wn(nn.Conv2d(in_channels=ch_last_upscale, out_channels=32 * final_scale**2, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.PixelShuffle(final_scale),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv_output = nn.Conv2d(in_channels=32, out_channels=out_nc, kernel_size=9, stride=1, padding=4, bias=True)
        
    def forward(self, input, depthMap, depthMask):
        # B, C, H, W = input.size() 
        # Encoder 
        if self.ablate_depth_matrix:
            feat_downscaled, depthVec = self.encoder(input )#,depthMask)
        else:
            feat_downscaled, depthVec = self.encoder(input ,depthMask)
        # Head
        fea_bef = self.head(feat_downscaled) 
        fea_in = fea_bef
        # Depth guided blocks
        for i in range(self.num_blocks-3):
            if i in self.which_ResBlk_depth:
                fea_in = self.__getattr__('depth-residual' + str(i+1))(fea_in, depthMap, depthMask, depthVec)
            else:
                fea_in = self.__getattr__('classic-residual' + str(i + 1))(fea_in)
        
        fea_mid = fea_in

        feat_add1 = torch.add(fea_mid, fea_bef)
        feat_up1 = self.upscale1(feat_add1) if self.scale == 8 else feat_add1
        
        if self.num_blocks-2 in self.which_ResBlk_depth:
            feat_up1 = self.__getattr__('depth-residual' + str(self.num_blocks - 1))(feat_up1, depthMap, depthMask, depthVec)
        else:
            feat_up1 = self.__getattr__('classic-residual' + str(self.num_blocks - 1))(feat_up1)
        
        feat_up2 = self.upscale2(feat_up1) if self.scale >= 4 else feat_up1

        if self.num_blocks-1 in self.which_ResBlk_depth:
            feat_up2 = self.__getattr__('depth-residual' + str(self.num_blocks))(feat_up2, depthMap, depthMask, depthVec)
        else:
            feat_up2 = self.__getattr__('classic-residual' + str(self.num_blocks))(feat_up2)

        feat_up3 = self.upscale3(feat_up2)
        
        out = self.conv_output(feat_up3)

        return torch.clamp(out, min=self.min, max=self.max)




class SFTMD(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, scale=4, input_para=10, min=0.0, max=1.0):
        super(SFTMD, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.num_blocks = nb

        self.conv1 = nn.Conv2d(in_nc, 64, 3, stride=1, padding=1)
        self.relu_conv1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.relu_conv2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        
        #sft_branch is not used.
        sft_branch = []
        for i in range(nb):
            sft_branch.append(SFT_Residual_Block())
        self.sft_branch = nn.Sequential(*sft_branch)


        for i in range(nb):
            self.add_module('SFT-residual' + str(i + 1), SFT_Residual_Block(nf=nf, para=input_para))

        self.sft = SFT_Layer(nf=64, para=input_para)
        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

        if scale == 4: #x4
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64 * scale, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64 * scale, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else: #x2, x3
            self.upscale = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64*scale**2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(scale),
            nn.LeakyReLU(0.2, inplace=True),
            )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=out_nc, kernel_size=9, stride=1, padding=4, bias=True)

    def forward(self, input, ker_code):
        B, C, H, W = input.size() # I_LR batch
        B_h, C_h = ker_code.size() # Batch, Len=10
        ker_code_exp = ker_code.view((B_h, C_h, 1, 1)).expand((B_h, C_h, H, W)) #kernel_map stretch

        fea_bef = self.conv3(self.relu_conv2(self.conv2(self.relu_conv1(self.conv1(input)))))
        fea_in = fea_bef
        for i in range(self.num_blocks):
            fea_in = self.__getattr__('SFT-residual' + str(i + 1))(fea_in, ker_code_exp)
        fea_mid = fea_in
        #fea_in = self.sft_branch((fea_in, ker_code_exp))
        fea_add = torch.add(fea_mid, fea_bef)
        fea = self.upscale(self.conv_mid(self.sft(fea_add, ker_code_exp)))
        out = self.conv_output(fea)

        return torch.clamp(out, min=self.min, max=self.max)


class SFTMD_DEMO(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, scale=4, input_para=10, min=0.0, max=1.0):
        super(SFTMD_DEMO, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.reses = nb

        self.conv1 = nn.Conv2d(in_nc + input_para, 64, 3, stride=1, padding=1)
        self.relu_conv1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.relu_conv2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        for i in range(nb):
            self.add_module('SFT-residual' + str(i + 1), SFT_Residual_Block(nf=64, para=input_para))

        self.sft_mid = SFT_Layer(nf=nf, para=input_para)
        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.scale = scale
        if scale == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif scale == 3:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64*9, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(3),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif scale == 2:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=out_nc, kernel_size=9, stride=1, padding=4, bias=False)

    def forward(self, input, code, clip=False):
        B, C, H, W = input.size()
        B, C_l = code.size()
        code_exp = code.view((B, C_l, 1, 1)).expand((B, C_l, H, W))

        input_cat = torch.cat([input, code_exp], dim=1)
        before_res = self.conv3(self.relu_conv2(self.conv2(self.relu_conv1(self.conv1(input_cat)))))

        res = before_res
        for i in range(self.reses):
            res = self.__getattr__('SFT-residual' + str(i + 1))(res, code_exp)

        mid = self.sft_mid(res, code_exp)
        mid = F.relu(mid)
        mid = self.conv_mid(mid)

        befor_up = torch.add(before_res, mid)

        uped = self.upscale(befor_up)

        out = self.conv_output(uped)
        return torch.clamp(out, min=self.min, max=self.max) if clip else out
