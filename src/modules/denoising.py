from importlib import import_module
import torch
from .loss import Loss
from ..data import tensor_split

class BasicModule:
    def __init__(self, cfg):
        self.model = self.make_model(cfg.MODEL).cuda()
        self.loss_function = self.make_loss(cfg)
        if cfg.TRAIN.PARALLE:
            self.model = torch.nn.DataParallel(self.model)
            
    def make_loss(self, cfg):
        return Loss(loss_fn=cfg.TRAIN.LOSS_TYPES, loss_fn_weight=cfg.TRAIN.LOSS_WEIGHTS)

    def make_model(self, cfg):
        module = import_module(f'src.models.{cfg.MODEL_NAME.lower()}')
        model = module.make_model(cfg)
        return model
    
    def forward(self, x):
        return self.model(x)

    def train_step(self, batch):
        image_target, image_noisy, sigma = [i.cuda() for i in batch]
        image_predict = self.forward(image_noisy)
        loss = self.loss_function(image_target, image_predict)
        return loss, image_target, image_predict, image_noisy
    
    def val_step(self, batch):
        self.model.eval()
        image_target, image_noisy, sigma = [i.cuda() for i in batch]
        image_predict = self.forward(image_noisy)
        return image_target, image_predict, image_noisy

class Module(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)