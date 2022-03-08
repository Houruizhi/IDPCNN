import os
import torch
import torch.nn as nn
from .ssim import SSIM
class AbsMSE(nn.Module):
    def __init__(self, *args, **kargs):
        super().__init__() 
        self.loss = nn.MSELoss(*args, **kargs)
    def forward(self, x, y):
        return self.loss((x.pow(2).sum(dim=1)+1e-11).sqrt(), y.pow(2).sum(dim=1).sqrt())

class Loss(nn.Module):
    def __init__(self, loss_fn, loss_fn_weight):
        super().__init__()
        self.loss_funcs = []
        loss_fn_names = loss_fn.split('+')
        loss_fn_weight = loss_fn_weight
        assert len(loss_fn_names) == len(loss_fn_weight)
        for weight, loss_type in zip(loss_fn_weight, loss_fn_names):
            if loss_type.lower() == 'l1':
                fn_ = nn.L1Loss(reduction='sum')
            elif loss_type.lower() == 'l2':
                fn_ = nn.MSELoss(reduction='sum')
            elif loss_type.lower() == 'l2abs':
                fn_ = AbsMSE(reduction='sum')
            elif loss_type.lower() == 'l2absmean':
                fn_ = AbsMSE(reduction='mean')
            elif loss_type.lower() == 'l2mean':
                fn_ = nn.MSELoss(reduction='mean')
            elif loss_type.lower() == 'l1mean':
                fn_ = nn.L1Loss(reduction='mean')
            elif loss_type.lower() == 'ssim':
                fn_ = SSIM(window_size=11)
            else:
                raise NotImplementedError
            
            self.loss_funcs.append([weight, fn_])
    
    def forward(self, target, tensor):
        loss = 0
        for weight, func in self.loss_funcs:
            loss_i = func(target, tensor)
            loss = loss + weight * loss_i
        return loss
