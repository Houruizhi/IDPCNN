from importlib import import_module
from .loss import Loss

def make_module(cfg):
    module = import_module(f'src.modules.{cfg.MODULE.NAME.lower()}')
    return module.Module(cfg)
