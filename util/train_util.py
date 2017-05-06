import torch
import numpy as np
import torch.nn as nn

# gernerate samples
def create_netG_indeps_sample(netG_indep, input, condition=0):
    '''create netG_indep_sample by netG in netG_indep

    in v1.0, condition for netG, not support now.

    - Params:
    @netG_indep: a list of netG
    @input: noise input from netG_share outputs
    @condition: condition for netG create sample

    - Returns:
    a list of samples created by netG_indep
    '''
    G_Indep_sample = []
    for net in netG_indep:
        G_Indep_sample.append(net(input))
    return G_Indep_sample


# computing loss
def netD_fake(indep_samples, netD):
    '''import fake samples, computing those fake prop

    - Params:
    @indep_samples: fake samples from netG

    - Returns:
    prop of those fake samples
    '''
    fake_prop = []
    for sample in indep_samples:
        fake_prop.append(netD(sample))
    return fake_prop

def compute_fake_loss(fake_prop, label):
    '''compute loss of netG by offical funcs

    - Params:
    @fake_prop: the prop of fake picture(sample created by netG)
    @label: BCEloss's label

    - Returns:
    the loss of netG
     '''
    fake_losses = []
    entropy = nn.BCELoss()
    for fake in fake_prop:
         fake_loss  = entropy(fake, label)
         fake_losses.append(fake_loss)
    return fake_losses

def find_best_netG(fake_prop):
    '''v1.0 find the best netG from netG_indep by its max netG_prop

    - Params:
    @fake_prop: the prop of fake imgs

    - Returns:
    the index of the best netG in neG_indep
    '''
    fake_losses = np.array([(torch.mean(fake)).data.numpy()[0] for fake in fake_prop])
    return np.argmax(fake_losses)

def compute_dloss(real_prop, fake_prop, label):
    '''v1.0 compute loss of netG and netD by offical funcs

    take the best-prop fake_prop as real-like prop, the real-like prop and real prop as real prop
    the rest of fake_prop as fake_prop

    - Params:
    @real_prop: the prop of dis real imgs
    @fake_prop: the prop of dis fake imgs
    @label: BCEloss's label

    - Returns: 
    the loss of 
    netD: log(D(x)) + log(1 - D(G(z)))
    '''
    netG_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    best_netG_index = find_best_netG(fake_prop)
    real_like_prop = fake_prop[best_netG_index]
    entropy = nn.BCELoss()

    label.data.fill_(0)
    fake_losses = compute_fake_loss(fake_prop, label)
    label.data.fill_(1)
    real_true_loss = entropy(real_prop, label)
    label.data.resize_(1).fill(1)
    real_like_loss = entropy(real_like_prop, label)
    rest_fake_loss = (sum(fake_losses[i] for i in netG_num) - fake_losses[best_netG_index]) / (len(netG_num) - 1)
    real_loss = real_true_loss + real_like_loss + rest_fake_loss

    return real_loss, real_true_loss, real_like_loss, rest_fake_loss

def compute_gloss(fake_prop, label):
    '''compute loss of netG by compute_fake_loss funcs

    - Params:
    @fake_prop: the prop of fake picture(sample created by netG)
    @label: BCEloss's label

    - Returns:
    the loss of 
    netG: log(D(G(z)))
     '''

    label.data.fill_(1)
    best_netG_index = find_best_netG(fake_prop)
    fake_losses = compute_fake_loss(fake_prop, label)
    return fake_losses, best_netG_index

# compute-loss cyclegan

# backward&step
def mutil_backward(netG_losses, net_share, net_indeps, index=None):
    '''mutil  backward() for netG_losses, let netG_losses[index].backward() as lastOne
       ... netG_share will backward only followed by netG_losses[index]

    - Params:
    @netG_losses: netG_losses
    @index: netG_losses[index] out of backward()
    '''
    for i in range(len(netG_losses)):
        if i == index:
            continue
        net_share.zero_grad()
        net_indeps[i].zero_grad()
        netG_losses[i].backward(retain_variables = True)

    if index != None:
        net_share.zero_grad()
        net_indeps[index].zero_grad()
        netG_losses[index].backward(retain_variables = True)     

def mutil_steps(netG_losses, net_share, net_indeps, index=None):
    '''v1.0 mutil step() for mutil net_solver

    - Params: 
    @netG_losses: loss for netG
    @net_indeps: mutil independly netG, each netG is net_indep
    @net_share: shared netG
    @index: net_indeps[index] be the lastOne to step()

    - Returns:
    no returns
    '''
    mutil_backward(netG_losses, net_share, net_indeps, index)
    for i in range(len(net_indeps)):
        if  i == index:
            continue
        net_indeps[i].step()
    if index != None:
        net_indeps[index].step()
        net_share.step()

# pre-process data
def link_data(data, times, dim):
    '''expand the data

    - Params:
    @data: the data flow in netG
    @dim: the dim-index of data
    @times: torch.cat([data, data]) times's times by the order of dim

    - Returns:
    dim = 1, time =3, the dim of data = (64, 1, 28, 28)
    return the dim of data = (64, 1*(times + 1), 28, 28)
    '''
    temp = data
    for i in range(times):
        data = torch.cat([data, temp], dim)
    return data    

def draft_data(fake_samples, cuda):
    '''if cuda is true, draft data from GPU. Otherwise, change nothing

    - Params:
    @fake_samples: netG(z)
    @cuda: true or false

    - Return
    a floattensor
    '''
    if cuda:
        return fake_samples.data.cpu()
    else:
        return fake_samples
