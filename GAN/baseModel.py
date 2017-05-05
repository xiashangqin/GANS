import os
import torch
import torchvision.utils as vutils

from G.netG import build_netG
from D.netD import build_netD

class _baseModel(object):
    '''Base Model combine netG and netD to became a gans's model

    - Attributes:
    @opt: options for config gans'model
    @train: train or test
    @x_dim: channels of input data
    @z_dim: dim of noise
    @g_model: what kind netG like, in cfg.py
    @d_model:what kind netD like, in cfg.py
    @netG: G
    @netD: D
    '''

    def __init__(self, opt):
        self.opt = opt
        self.train = opt.train
        self.x_dim = opt.x_dim
        self.z_dim = opt.z_dim
        self.g_model = opt.g_model
        self.d_model = opt.d_model
        self.netG = build_netG(opt.g_model, opt.z_dim)
        self.netD = build_netD(opt.d_model, opt.x_dim)

    def __str__(self):
        netG = self.netG.__str__()
        netD = self.netD.__str__()
        return 'Gan:\n' + '{}{}'.format(netG, netD)

    def train(self):
        '''train gans
        '''
        pass

    def test(self):
        '''test gans
        '''
        pass

    def save_network(self, it, savepath):
        '''save checkpoints of netG and netD in savepath

        - Params:
        @it: number of iterations
        @savepath: in savepath, save network parameter
        '''
        torch.save(self.netG.state_dict(), '%s/netG_epoch_%d.pth' % (savepath, it))
        torch.save(self.netD.state_dict(), '%s/netD_epoch_%d.pth' % (savepath, it))
    
    def load_network(self, g_network_path):
        '''load network parameters of netG and netD

        - Params:
        @g_network_path: the path of netG
        '''
        self.netG.load_state_dict(torch.load(g_network_path))

    def save_image(self, fake, it , savepath):
        '''save result of netG output

        - Params:
        @fake: the output of netG
        @it: number of iterations
        @savepath: in savepath, save network parameter
        '''
        vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (savepath, it))


        

