import torch.nn as nn
import math
def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

def weights_init_uniform(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        m.weight.data.uniform_(-1, 1)
        # nn.init.orthogonal_(m.weight.data, gain=0.2)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d):
        # nn.init.uniform_(m.weight.data, 0.1, 1.0)
        # nn.init.constant_(m.bias.data, 0.0)
        m.weight.data.uniform_(-1, 1)
        nn.init.constant_(m.bias.data, 0.0)
