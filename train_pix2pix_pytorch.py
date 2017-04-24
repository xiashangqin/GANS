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

# loading datasets
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data/torch_mnistdata', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True, **{})

# global setting
mb_size = 64
z_dim = 100
h_dim = 128
x_ dim_w, x_dim_h =train_loader.dataset.train_data.size()[1:3] 
x_dim = x_dim_w*x_dim_h
train_size = train_loader.dataset.train_data.size()[0]
y_dim = 10
lr = 1e-3
cnt = 0
nets_num = 10

cuda = False
niter = 25

# build gans
netD = build_netD(config['D'][3], x_dim)
netG = build_netG(config['G'][4], z_dim)

# init gans
netD.apply(weight_init)
netG.apply(weight_init)

# build gans's solver
G_solver = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
D_solver = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

# announce input
x = torch.FloatTensor(mb_size, 3, x_dim_w, x_dim_h)
z = torch.FloatTensor(mb_size, z_dim, 1, 1)

# init input in cuda, then convert floattensor to variable
if cuda:
    netD.cuda()
    netG.cuda()
    x, z = x.cuda(), z.cuda()

x = Variable(x)
z = Variable(z)

# training
for it in range(niter):
    for batch_idx, (data, target) in enumerate(train_loader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        netD.zero_grad()
        x.data.resize_(data.size()).copy_(data)
        z.data.resize_(mb_size, z_dim, 1, 1).normal(0, 1)

        D_real = netD(x)
        D_fake = netD(z)

        D_loss = -(torch.mean(torch.log(D_real)) + torch.mean(torch.log(1 - D_fake)))
        D_loss.backward()
        D_solver.step()

        
        




