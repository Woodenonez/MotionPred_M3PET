import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_
from net_module.backbones import *

from util.datatype import *
import matplotlib.pyplot as plt


class UNetPlain(nn.Module):
    '''
    Description
        :A plain/original UNet implementation.
    Comment
        :The input size is (batch x channel x height x width).
    '''
    def __init__(self, in_channels, num_classes=1, with_batch_norm=True, bilinear=True, lite:bool=True):
        super(UNetPlain,self).__init__()
        self.unet = UNet(in_channels, num_classes, with_batch_norm, bilinear, lite=lite)

    def forward(self, x):
        logits = self.unet(x)
        return logits

class UNetPos(nn.Module):
    '''
    Description
        :A modified UNet implementation with an output layer that only outputs positive values.
        :The output layer can be 'softplus', 'poselu'.
    '''
    def __init__(self, in_channels, num_classes=1, with_batch_norm=True, bilinear=True, lite:bool=True, out_layer:str='softplus'):
        super(UNetPos,self).__init__()
        if out_layer.lower() not in ['softplus', 'poselu']:
            raise ValueError(f'The output layer [{out_layer}] is not recognized.')
        self.unet = UNet(in_channels, num_classes, with_batch_norm, bilinear, lite=lite)
        if out_layer == 'softplus':
            self.outl = torch.nn.Softplus()
        elif out_layer == 'poselu':
            self.outl = PosELU(1e-6)

    def forward(self, x):
        logits = self.unet(x)
        return self.outl(logits)

class E3Net(nn.Module): # 
    '''
    Ongoing, the idea is to have an Early Exit Energy-based (E3) network.
    
    Comment
        :The input size is (batch x channel x height x width).
    '''
    def __init__(self, in_channels, en_channels, de_channels, num_classes=1, with_batch_norm=False, out_layer:str='softplus'):
        super(E3Net,self).__init__()
        if (out_layer is not None): 
            if (out_layer.lower() not in ['softplus', 'poselu']):
                raise ValueError(f'The output layer [{out_layer}] is not recognized.')

        self.encoder = UNetTypeEncoder(in_channels, en_channels, with_batch_norm)
        self.inc = DoubleConv(en_channels[-1], out_channels=en_channels[-1], with_batch_norm=with_batch_norm)

        up_in_chs  = [en_channels[-1]] + de_channels[:-1]
        up_out_chs = up_in_chs # for bilinear
        dec_in_chs  = [enc + dec for enc, dec in zip(en_channels[::-1], up_out_chs)] # add feature channels
        dec_out_chs = de_channels

        self.decoder = nn.ModuleList()
        for in_chs, out_chs in zip(dec_in_chs, dec_out_chs):
            self.decoder.append(UpBlock(in_chs, out_chs, bilinear=True, with_batch_norm=with_batch_norm))
        
        self.multi_out_cl = nn.ModuleList() # out conv layer
        for de_ch in de_channels:
            self.multi_out_cl.append(nn.Conv2d(de_ch, num_classes, kernel_size=1))

        if out_layer == 'softplus':
            self.outl = torch.nn.Softplus()
        elif out_layer == 'poselu':
            self.outl = PosELU(1e-6)
        else:
            self.outl = torch.nn.Identity()

    def forward(self, x):
        features:list = self.encoder(x)

        features = features[::-1]
        x = self.inc(features[0])

        multi_out = []
        for feature, dec, out in zip(features[1:], self.decoder, self.multi_out_cl):
            x = dec(x, feature)
            multi_out.append(self.outl(out(x)))
        return multi_out


