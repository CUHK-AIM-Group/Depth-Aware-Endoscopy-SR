
from torch import nn
import torch
from torchvision import models

def choose_vgg(name):

    f = None

    if name == 'vgg11':
        f = models.vgg11(pretrained = True)
    elif name == 'vgg11_bn':
        f = models.vgg11_bn(pretrained = True)
    elif name == 'vgg13':
        f = models.vgg13(pretrained = True)
    elif name == 'vgg13_bn':
        f = models.vgg13_bn(pretrained = True)
    elif name == 'vgg16':
        f = models.vgg16(pretrained = True)
    elif name == 'vgg16_bn':
        f = models.vgg16_bn(pretrained = True)
    elif name == 'vgg19':
        f = models.vgg19(pretrained = True)
    elif name == 'vgg19_bn':
        f = models.vgg19_bn(pretrained = True)

    for params in f.parameters():
        params.requires_grad = False

    return f

pretrained_mean = torch.tensor([0.485, 0.456, 0.406], requires_grad = False).view((1, 3, 1, 1))
pretrained_std = torch.tensor([0.229, 0.224, 0.225], requires_grad = False).view((1, 3, 1, 1))

class VGGNet(nn.Module):

    def __init__(self, name, layers, device = None):

        super(VGGNet, self).__init__()
        self.vgg = choose_vgg(name)
        self.layers = layers

        features = list(self.vgg.features)[:max(layers) + 1]
        self.features = nn.ModuleList(features).eval()

        self.mean = pretrained_mean.to(device)
        self.std = pretrained_std.to(device)

    def forward(self, x, retn_feats=None, layers=None):

        x = (x - self.mean) / self.std

        results = []

        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in self.layers:
                results.append(x.view(x.shape[0], -1))

        return results