# Small Project for me
> just gan!!!!


## Works
What should we do? And how I do?

### Pre-works

* [x] -nets
    * [x] - move create_nets to G.netG
* [x] - solver
    * [x] - create mutil solvers for mutil netG_share+netG_indep
    * [x] - create mutil solvers for mutil netG_indep
    * [x] - create solver for netG_share
    * [x] - create solver for mutil netD
* [ ] - train
    * [x] - init network's weights and bias of netG and netD
    * [x] - compute prop of fake for each fake sample
    * [ ] - find out best netG

### v1.0

* no condition version.
* each netG rise in a competition, and the best netG limits the rest of the network
* **use netG best-one and real-data as real-prop, and others are fake-prop**

### v1.1

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

get troubled in computing netD_loss