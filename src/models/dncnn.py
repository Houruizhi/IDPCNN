import torch
import torch.nn as nn
from .weight_init import weights_init_kaiming

def make_model(cfg):
    return DnCNN(
        in_channels=cfg.IN_CHANNELS, 
        out_channels=cfg.OUT_CHANNELS, 
        num_of_layers=cfg.NUM_ITER, 
        num_features=cfg.NUM_FEATURES,
        residual=cfg.RESIDUAL,
        kernel_size=cfg.KERNEL_SIZE, 
        bn=cfg.BN, 
        bias=cfg.BIAS
        )
        
class DnCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_of_layers, num_features, kernel_size=3, residual=True, bn=True, bias=False):
        super(DnCNN, self).__init__()
        self.residual = residual
        if self.residual:
            assert in_channels == out_channels
        kernel_size = kernel_size
        padding = kernel_size//2
        features = num_features
        in_channels = in_channels
        groups = 1
        layers = []
        layers.append(nn.ReflectionPad2d((padding,padding,padding,padding)))
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=features, 
            kernel_size=kernel_size, bias=bias))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.ReflectionPad2d((padding,padding,padding,padding)))
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, 
                kernel_size=kernel_size, bias=bias, groups=groups))
            if bn:
                layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ReflectionPad2d((padding,padding,padding,padding)))
        layers.append(nn.Conv2d(in_channels=features, out_channels=out_channels, 
            kernel_size=kernel_size, bias=bias))
        self.dncnn = nn.Sequential(*layers)
        self.dncnn.apply(weights_init_kaiming)
    def forward(self, inputs, sigma=None):
        x = self.dncnn(inputs)
        if self.residual:
            return inputs - x
        else:
            return x

