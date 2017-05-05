# updates for mygans
> gans更新记录

## Updates - v1.0
Recode updates everyday.

* **created**: create files from 0-1
* **add**: add new to origin
* **changed**: change somethings
* **move**: move files/funcs from path1 to path2
* **fixed**: fix bugs

### 2017-4-10

* in ./G/netG.py - **add** note for def
* in ./util/network_util.py - **add** def create_nets
* in ./train.py - **create** mutil nets by using create_nets
* in README.md - **create** this markdown file

### 2017-4-11

* in ./util/network_util.py - **add** def create_optims and create_couple_optims
* in ./util/network_util.py - **add** def init_networks
* in ./util/solver_util.py - **create** this file, and **move** create_optims and create_couple_optims to this file
* in ./util/train_util.py - **create** this file, and **add** def create_netG_indeps and netD_fake
* in ./train.py - **add** G_share_solver, G_indep_solver, G_solvers

### 2017-4-12

* in ./util/train_util.py - **add** def find_best_netG and def compute_loss
* in ./util/train_util.py - **change** def create_netG_indeps to def create_netG_samples
* in ./util/network_util.py - **change** def weight_init to suit mutil kinds of netG

### 2017-4-14

**tips:**pytorch-tips updates showed in [Pytorch-learnBooks](https://github.com/JiangWeixian/Pytorch-LearnBooks)

* in ./util/solver_util.py - **add** def create_couple2one_optims

### 2017-4-17

G_solvers : mutil (netG_indep_solver + netG_share_solver) not be used

* in ./util/train_util.py - **add** def mutil_backward
* in ./util/train_util.py - **add** def mutil_steps
* in ./train.py - **add** plt.imshow and net.save, **add** iteration

### 2017-4-18

use totchvision in pytorch is better, so **changed** load data way!

* in ./train_mnist_tf.py - **changed** train.py to train_mnist_tf.py
* in ./train_mnist_pytorch.py - **add** train_mnist_pytorch.py

### 2017-4-22

find some bugs.

* in ./util/vision_util.py&./util/train_util.py, ./train_mnist_pytorch.py&train_mnist_tf.py - **fixed** bugs
* bugs - init network as same mean&var will be better

but the problem is that the index of best netG didn't changed over training! So, the best netG_indep always is best, don't change anymore! 

Because of init?

### 2017-4-24

I find that gans's examples in pytorch, always netD.step(), then calculate the prob of fake again, then netG.step().In other words, gan's examples calculate the prob of fake twice. But in branch master, I always calculate the fake only once.So

* in ./util/train_util.py - **changed** the order of step() and backward() in def mutil_backward&mutil_steps
* in ./train_mnist_pytorch.py - **changed** calculate netG's loss twice

### 2017-4-25

* ./train_mnist_pytorch.py - **add** z.data.resize_(mb_size, z_dim).normal_(0, 1) in if it % 2 ==0:

### 2017-4-26

* in train_mnist_pytorch.py - this project don't work

### 2017-4-27

* in train_mnist_pytorch.py - **changed** batch-size=64 to 1
* in train_util.py - **change** log(1 - D(G(z))) to BCEloss in **def compute_dloss and compute gloss**

But still not working!

## Updates - v1.1
Recode updates everyday.

* **created**: create files from 0-1
* **add**: add new to origin
* **changed**: change somethings
* **move**: move files/funcs from path1 to path2
* **fixed**: fix bugs

### 2017-4-24

* in ./D/cfg.py&./D/netD.py and ./G/cfg.py&./G/netG.py - **add** def create_convnets_G&create_convnets_D to bulid dcgans

I find that gans's examples in pytorch, always netD.step(), then calculate the prob of fake again, then netG.step().In other words, gan's examples calculate the prob of fake twice. But in branch master, I always calculate the fake only once.

### 2017-4-25

* ./train_pix2pix_pytorch.py -**add** x.data.resize_... to suit mnist datasets, and crayon
* ./train_pix2pix_pytorch.py - **change** def weight_init to suit conv's network

### 2017-4-26

the code(normalized in dataloader) will destory training! So just remove it in all files! **and** we try my ten coupled gans after that!

* ./train_vanillgan_pytorch.py&./train_vanillgan_pytorch.py - **created** those files, complete training vanillgans, test gans in load_state_stict()
* ./train_chanierdcgan_mnist_pytorch.py&./train_dcgan_mnist_pytorch.py - **created** those files, want train mnist datasets in dcgans
* ./test_mnist_dcgan.py - **created** this file, and netG and netD based on chanier-gans
* ./D/cfg.py&./G/cfg.py - **add** dcgans, chainer-gans's arch

### 2017-4-28

* in ./train_chanierdcgan_mnist_pytorch.py - it does work for mnist! 
* in ./train_dcgan_mnist_pytorch.py - **delete** this file, it doesn't work

### 2017-5-5

Use nn.BCEloss, rather than **-(torch.mean(torch.log(D_real)) + torch.mean(torch.log(1 - D_fake)))** Somethimes,  the after One will have a mistake! So use offical loss funcs!

