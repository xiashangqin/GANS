import torch
import math
import numpy as np
from G.netG import _netG
from D.netD import _netD
from G.netG import build_netG

def weight_init(m):
    '''
    init net weights and bias for linear-layer
    '''
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Conv') !=-1:
        m.weight.data.normal_(0, 0.02)

def create_nets(config, input_dim, num):
    '''create muti netG, and append into nets.

    - Params:
    @config: config for netG
    @input_dim: the dim of input
    @num: num of netG

    - Returns
    mutils nets
    '''
    nets = []
    for i in range(num):
        netG = build_netG(config[3], input_dim)
        nets.append(netG)
    return nets

def init_network(nets):
    '''def for init netG or netD, it's a callback func

    - Params:
    @nets: _netG type for weight_init; other types call init_network again util nets belong to _netG
    '''
    if type(nets) == _netG or type(nets) == _netD:
        nets.apply(weight_init)
    else:
        for net in nets:
            init_network(net)


if __name__  == '__main__':
    create_nets(['fc', 'fc_competition', 'fc_cs', 'fc_ci'], 100, 10)