from .dncnn import DnCNN
from .unet import Unet

from importlib import import_module

def make_model(cfg):
    module = import_module(f'src.models.{cfg.MODEL_NAME.lower()}')
    model = module.make_model(cfg)
    return model
    