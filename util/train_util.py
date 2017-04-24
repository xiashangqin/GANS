import torch
import numpy as np

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

def compute_fake_loss(fake_prop):
    '''compute loss of netG

    - Params:
    @fake_prop: the prop of fake picture(sample created by netG)

    - Returns:
    the loss of netG
     '''
    fake_losses = []
    for fake in fake_prop:
         fake_loss  = -torch.mean(torch.log(1 - fake))
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

def compute_loss(real_prop, fake_prop):
    '''v1.0 compute loss of netG and netD

    take the best-prop fake_prop as real-like prop, the real-like prop and real prop as real prop
    the rest of fake_prop as fake_prop

    - Params:
    @real_prop: the prop of dis real imgs
    @fake_prop: the prop of dis fake imgs

    - Returns: 
    the loss of netD and netG
    '''
    netG_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    best_netG_index = find_best_netG(fake_prop)
    real_like_prop = fake_prop[best_netG_index]

    fake_losses = compute_fake_loss(fake_prop)
    real_true_loss = -torch.mean(torch.log(real_prop))
    real_like_loss = -torch.mean(torch.log(real_like_prop))
    rest_fake_loss = (sum(fake_losses[i] for i in netG_num) - fake_losses[best_netG_index]) / (len(netG_num) - 1)
    real_loss = real_true_loss + real_like_loss + rest_fake_loss

    return real_loss, fake_losses, best_netG_index

def mutil_backward(netG_losses, net_share, index=None):
    '''mutil  backward() for netG_losses, let netG_losses[index].backward() as lastOne
       ... netG_share will backward only followed by netG_losses[index]

    - Params:
    @netG_losses: netG_losses
    @index: netG_losses[index] out of backward()
    '''
    for i in range(len(netG_losses)):
        if i == index:
            continue
        netG_losses[i].backward(retain_variables=True)
        net_share.zero_grad()

    if index != None:
        netG_losses[index].backward(retain_variables=True)     

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
    mutil_backward(netG_losses, net_share, index)
    for i in range(len(net_indeps)):
        if  i == index:
            continue
        net_indeps[i].step()
        net_indeps[i].zero_grad()
    if index != None:
        net_indeps[index].step()
        net_share.step()
        net_indeps[index].zero_grad()
        net_share.zero_grad()

