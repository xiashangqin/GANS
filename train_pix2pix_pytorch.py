import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as nn
import torch.optim as optim
from pycrayon import CrayonClient
from torch.autograd import Variable

from cfg import *
from D.netD import build_netD
from G.netG import build_netG
from torchvision import datasets, transforms
from util.network_util import create_nets, init_network, weight_init
from util.solver_util import create_couple2one_optims, create_optims
from util.train_util import (compute_loss, create_netG_indeps_sample,
                             mutil_steps, netD_fake)
from util.vision_util import (add2experiments, create_experiments,
                              create_sigle_experiment)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data/torch_mnistdata', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True, **{})
mb_size = 64
z_dim = 100
h_dim = 128
x_dim = train_loader.dataset.train_data.size()[1]*train_loader.dataset.train_data.size()[2]
train_size = train_loader.dataset.train_data.size()[0]
y_dim = 10
lr = 1e-3
cnt = 0
nets_num = 10

netD = build_netD(config['D'][3], x_dim)




