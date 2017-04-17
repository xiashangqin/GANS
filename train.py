import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as nn
import torch.optim as optim
from pycrayon import CrayonClient
from tensorflow.examples.tutorials.mnist import input_data
from torch.autograd import Variable

from cfg import *
from D.netD import build_netD
from G.netG import build_netG
from util.network_util import create_nets, init_network, weight_init
from util.solver_util import create_couple2one_optims, create_optims
from util.train_util import (compute_loss, create_netG_indeps_sample,
                             mutil_steps, netD_fake)

mnist = input_data.read_data_sets('./mnist_data/data', one_hot=True)
mb_size = 64
z_dim = 100
h_dim = 128
x_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
lr = 1e-3
cnt = 0
nets_num = 10

'''
cc = CrayonClient(hostname="localhost")
cc.remove_all_experiments()
try:
    G_exp = cc.open_experiment('G_loss')
except Exception, e:
    G_exp = cc.create_experiment('G_loss')

try:
    D_exp = cc.open_experiment('D_loss')
except Exception, e:
    D_exp = cc.create_experiment('D_loss')

try:
    Prob = cc.open_experiment('Prob')
except Exception, e:
    Prob = cc.create_experiment('Prob')
'''


netG_share = build_netG(config['G'][2], z_dim)
netG_indeps = create_nets(config['G'], h_dim, nets_num)
netD = build_netD(config['D'][2], x_dim)

init_network(netG_share)
init_network(netG_indeps)
init_network(netD)

D_solver = optim.Adam(netD.parameters(), lr=lr)
G_share_solver = optim.Adam(netG_share.parameters(), lr=lr)
G_indep_solver = create_optims(netG_indeps, [lr,])
G_solvers = create_couple2one_optims(netG_share, netG_indeps, [lr,])


'''
params = [item.size() for item in list(netG.parameters())]
print params
'''

z = Variable(torch.randn(mb_size, z_dim))
X, c = mnist.train.next_batch(mb_size)
X = Variable(torch.from_numpy(X))
c = Variable(torch.from_numpy(c.astype('float32')))

G_share_sample = netG_share(z)
G_indep_sample = create_netG_indeps_sample(netG_indeps, G_share_sample)

D_real = netD(X)
D_fake = netD_fake(G_indep_sample, netD)
D_loss, G_losses, index = compute_loss(D_real, D_fake)

print list(netG_share.parameters())

D_loss.backward(retain_variables=True)
D_solver.step()
D_solver.zero_grad()

mutil_steps(G_losses, G_share_solver, G_indep_solver, index)
