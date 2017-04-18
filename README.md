# Small Project for me
> just gan!!!!


## Works
What should we do? And how I do?

### Pre-works

* [x] -nets
    * [x] - move create_nets to G.netG
    * [x] - init different weights and bias for different netG
* [x] - solver
    * [x] - create mutil solvers for mutil netG_share+netG_indep
    * [x] - create mutil solvers for mutil netG_indep
    * [x] - create solver for netG_share
    * [x] - create solver for mutil netD
* [ ] - train
    * [x] - init network's weights and bias of netG and netD
    * [x] - compute prop of fake for each fake sample
    * [x] - find out best netG

## v1.0

* no condition version.
* each netG rise in a competition, and the best netG limits the rest of the network
* **use netG best-one and real-data as real-prop, and others are fake-prop**

## Arch of Gans

[Arch of gans](https://github.com/JiangWeixian/GANS/tree/master/README/v1.0/noise-Z.png)

## v1.1

## Updates
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

**now, we can start training!** 

### 2017-4-18

use totchvision in pytorch is better, so **changed** load data way!

* in ./train_mnist_tf.py - **changed** train.py to train_mnist_tf.py
* in ./train_mnist_pytorch.py - **add** train_mnist_pytorch.py