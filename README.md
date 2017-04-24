# Small Project for me
> just gan!!!!

## v1.0

* no condition version.
* each netG rise in a competition, and the best netG limits the rest of the network
* **use netG best-one and real-data as real-prop, and others are fake-prop**

### Pre-works - v1.0

* [x] -nets
    * [x] - move create_nets to G.netG
    * [x] - init different weights
* [x] - solver
    * [x] - create mutil solvers for mutil netG_share+netG_indep
    * [x] - create mutil solvers for mutil netG_indep
    * [x] - create solver for netG_share
    * [x] - create solver for mutil netD
* [x] - train
    * [x] - init network's weights and bias of netG and netD
    * [x] - compute prop of fake for each fake sample
    * [x] - find out best netG

### Arch of Gans

![Arch of gans](https://github.com/JiangWeixian/GANS/blob/master/README/v1.0/noise-Z.png)

* Step1: any netG share low layer, and indepently higher layers. Generate samples as normal Gan's do
* Step2: netD distinguish true images and fake samples, then and the best netG by the higher prob of fake samples
* Step3: netG_share backward&step followed by best netG, netG_indep backward&step normaly

### Loss format

![netG_Loss](https://github.com/JiangWeixian/GANS/blob/master/README/v1.0/netG_loss.gif)

![netD_Loss](https://github.com/JiangWeixian/GANS/blob/master/README/v1.0/netD_loss.gif)



