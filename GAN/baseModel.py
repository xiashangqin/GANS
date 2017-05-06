import os
import torch
import torchvision.utils as vutils

from pycrayon import CrayonClient


class _baseModel(object):
    '''Base Model combine netG and netD to became a gans's model

    - Attributes:
    @opt: options for config gans'model
    @train: train or test
    @cc: crayon client or not
    @cuda: use cuda or not
    '''

    def __init__(self, opt):
        self.opt = opt
        self.train = opt.train
        self.cc = CrayonClient(hostname="localhost") if opt.cc else opt.cc
        self.cuda = opt.cuda

    def create_tensorboard(self):
        '''use docker create tensorboard
        '''
        pass
    

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


        

