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

## Updates - v1.1
Recode updates everyday.

* **created**: create files from 0-1
* **add**: add new to origin
* **changed**: change somethings
* **move**: move files/funcs from path1 to path2
* **fixed**: fix bugs

### 2017-4-24

* in ./D/cfg.py&./D/netD.py and ./G/cfg.py&./G/netG.py - **add** def create_convnets_G&create_convnets_D to bulid dcgans

I find that gans's examples in pytorch, always netD.step(), then calculate the prob of fake again, then netG.step().In other words, gan's examples calculate the prob of fake twice. But in branch master, I always calculate the fake only once.So

* in ./util/train_util.py - **changed** the order of step() and backward() in def mutil_backward&mutil_steps
* in ./train_mnist_pytorch.py - **changed** calculate netG's loss twice