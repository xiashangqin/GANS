import torch
import torch.nn as nn
from cfg import *

class _netD(nn.Module):
    def __init__(self, layers, ngpu = 0):
        super(_netD, self).__init__()
        self.main = nn.ModuleList(layers)
    
    def forward(self, input, condition = None):
        x = input
        c = condition
        for index in range(len(self.main)):
            if index in netNumDCConfig['fc_competition_out']:
                x = torch.cat([x, c], 1)
                x = self.main[index](x)
            else:
                x = self.main[index](x)
        return x


def create_convnets_D(cfg, x_dim = 0, c_dim = 0, batch_norm = False):
    layers = []
    i_dim = x_dim + c_dim
    for v in cfg:
        if v == 'R':
            layers += [nn.ReLU(inplace=True)]
        elif v == 'LR':
            layers += [nn.LeakyReLU(0.2, inplace=False)]
        elif v == 'S':
            layers += [nn.Sigmoid()]
        elif v == 'B':
            layers += [nn.BatchNorm2d(i_dim)]
        else:
            o_dim, k, s, p = v
            layers += [nn.Conv2d(i_dim, o_dim, kernel_size=k, stride=s, padding=p, bias=False)]
            i_dim = o_dim
    return layers


def build_netD(cfg_index, x_dim = 0, c_dim = 0, batch_norm=False):
    '''build sigle netD

    - Params
    @cfg_index: the index-label of structure of netD
    @x_dim: input-data dim
    @c_dim: condition dim, but not for netD

    - Returns
    the class netD
    '''
    network = create_convnets_D(netDNumConfig[cfg_index], x_dim, c_dim)
    return _netD(layers = network)

if __name__ == '__main__':
    network = create_convnets_D(netDNumConfig['dcgans'], x_dim = 3, c_dim=0)
    netD = _netD(layers = network)
    print netD