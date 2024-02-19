
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from torch.nn.parameter import Parameter
class SEAN(nn.Module):
    def __init__(self, label_nc=10, norm_nc=32, len_latent=256, inject_st=True, param_free_norm_type='instance', use_trainable_params=True, norm_gamma=0.1, norm_beta=0.1, ablate_depth_matrix=False, ablate_depth_block=False):
        super().__init__()
        ks = 3#int(parsed.group(2))
        self.len_latent = len_latent
        self.inject_st = inject_st
        self.ablate_depth_matrix = ablate_depth_matrix
        self.ablate_depth_block = ablate_depth_block

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)
        pw = ks // 2
        if self.inject_st:
            self.A_i_j = nn.Conv2d(label_nc, label_nc, kernel_size=1, padding=0)
            self.mlp_gamma_s = nn.Conv2d(self.len_latent, norm_nc, kernel_size=ks, padding=pw)
            self.mlp_beta_s = nn.Conv2d(self.len_latent, norm_nc, kernel_size=ks, padding=pw)
            if use_trainable_params:
                self.alpha_beta = Parameter(torch.rand(1), requires_grad=True)
                self.alpha_gamma = Parameter(torch.rand(1), requires_grad=True)
            else:
                self.alpha_beta = norm_beta
                self.alpha_gamma = norm_gamma
        nhidden = norm_nc * 2 #128
        self.mlp_mask = nn.Sequential(
            nn.Conv2d(1, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma_o = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta_o = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

        if self.ablate_depth_block:
            # downsample: 256x256 -> 128x128
            self.mlp_depthMatrix = nn.ConvTranspose2d(label_nc, label_nc, 3, stride=2, padding=1)
            self.mlp_before_all = nn.Conv2d(label_nc + nhidden, norm_nc, kernel_size=ks, padding=pw)
            self.mlp_gamma_all = nn.Conv2d(label_nc + nhidden, norm_nc, kernel_size=ks, padding=pw)
            self.mlp_beta_all = nn.Conv2d(label_nc + nhidden, norm_nc, kernel_size=ks, padding=pw)

    # def forward(self, st, segmap, x):
    def forward(self, x, depthMap, depthMask, st):

        assert self.len_latent == st.size(2) and st.size(1) == depthMask.size(1)

        normalized = self.param_free_norm(x)

        depthMap = F.interpolate(depthMap, size=x.size()[2:], mode='nearest')
        depthMask = F.interpolate(depthMask, size=x.size()[2:], mode='nearest')
        
        actv = self.mlp_mask(depthMap)
        # ablate the depth block to the general normalization
        if self.ablate_depth_block:
            # repeat depth vector
            duplicated_st = st.repeat(1,1,1,st.size(2)) # bsx10x256x256
            downscaled_st = self.mlp_depthMatrix(duplicated_st)
            concat_input = torch.cat([downscaled_st,actv], dim=1)
            concat_input = self.mlp_before_all(concat_input)
            gamma = self.mlp_gamma_all(concat_input)
            beta = self.mlp_beta_all(concat_input)
            out = normalized * (1 + gamma) + beta
        else:
            beta_o = self.mlp_beta_o(actv)
            gamma_o = self.mlp_gamma_o(actv)
            if self.inject_st:
                if self.ablate_depth_matrix:
                    beta_s = self.mlp_beta_s(st)
                    gamma_s = self.mlp_gamma_s(st)
                else:
                    st = self.A_i_j(st.unsqueeze(3))
                    st = st.expand(st.size(0), st.size(1), st.size(2), depthMask.size(3)).permute(0, 3, 2, 1)
                    style_map = st.matmul(depthMask.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

                    beta_s = self.mlp_beta_s(style_map)
                    gamma_s = self.mlp_gamma_s(style_map)

                gamma = self.alpha_gamma * gamma_s + (1. - self.alpha_gamma) * gamma_o
                beta = self.alpha_beta * beta_s + (1. - self.alpha_beta) * beta_o
                out = normalized * (1 + gamma) + beta
            else:
                out = normalized * (1 + gamma_o) + beta_o
        return out