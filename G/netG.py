import torch
import torch.nn as nn
from cfg import *

class _netG(nn.Module):
    def __init__(self, layers, ngpu = 0):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.ModuleList(layers)
    
    def forward(self, input = 0, condition = 0):
        z = input
        c = condition
        for index in range(len(self.main)):
            if index in netNumGCConfig['fc_competition_out']:
                z = torch.cat([z, c], 1)
                z = self.main[index](z)
            else:
                z = self.main[index](z)
        return z



def create_fcnets_G(cfg, z_dim = 0, c_dim = 0, batch_norm = False):
    layers = []
    i_dim = z_dim + c_dim
    for v in cfg:
        if v == 'R':
            layers += [nn.ReLU(inplace=True)]
        elif v == 'S':
            layers += [nn.Sigmoid()]
        else:
            if v[-1] == 'd':
                v = int(v[:-1])
                layers += [nn.Linear(i_dim, v, bias=True)]
                i_dim = v + c_dim
            else:
                v = int(v)
                layers += [nn.Linear(i_dim, v, bias=True)]
                i_dim = v

    return layers


def build_netG(cfg_index, z_dim = 0, c_dim = 0, batch_norm=False):
    '''build sigle netG
    - Params:
    @netGS: index in cfg, shared layers
    @netGI: index in cfg, independ layers
    @z_dim: nosie dim
    @c_dim: conditions dim
    @batch_norm: not support now

    - Returns:
    the class netG
    '''
    network=create_fcnets_G(netNumGConfig[cfg_index], z_dim, c_dim)
    return _netG(layers=network)


if __name__ == '__main__':
    network = create_fcnets_G(netNumGConfig['fc'], 100, 10)
    netG = _netG(layers = network)
    print netG