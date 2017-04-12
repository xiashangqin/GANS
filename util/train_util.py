import torch
import numpy as np

def create_netG_indeps(netG_indep, input, condition=0):
    '''create netG_indep_sample by netG in netG_indep

    in v1.0, condition for netG, not support now.

    - Params:
    @netG_indep: a list of netG
    @input: noise input from netG_share outputs
    @condition: condition for netG create sample

    - Returns:
    samples created by netG_indep
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

def compute_loss(real_prop, fake_prop):
    '''v1.0 compute loss of netG

    take the best-prop fake_prop as real-like prop, the real-like prop and real prop as real prop
    the rest of fake_prop as fake_prop

    - Params:
    @real_prop: the prop of dis real imgs
    @fake_prop: the prop of dis fake imgs

    - Returns: 
    the loss of netD
    '''
    print 'loss'

def find_best_netG(fake_prop):
    '''find the best netG from netG_indep by its netG_prop

    - Params:
    @fake_prop: the prop of fake imgs

    - Returns:
    the index of the best netG in neG_indep
    '''
    fake_losses = np.array([(torch.mean(fake)).data.numpy()[0] for fake in fake_prop])
    print fake_losses
    print np.argmax(fake_losses)


    
         
    