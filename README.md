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

### Future-works-v1.0

* [ ] modify batch size=1

### Arch of Gans

![Arch of gans](https://github.com/JiangWeixian/GANS/blob/master/README/v1.0/noise-Z.png)

* Step1: any netG share low layer, and indepently higher layers. Generate samples as normal Gan's do
* Step2: netD distinguish true images and fake samples, then and the best netG by the higher prob of fake samples
* Step3: netG_share backward&step followed by best netG, netG_indep backward&step normaly

### Loss format

![netG_Loss](https://github.com/JiangWeixian/GANS/blob/master/README/v1.0/netG_loss.gif)

![netD_Loss](https://github.com/JiangWeixian/GANS/blob/master/README/v1.0/netD_loss.gif)

## v1.1

v1.1 version of gans always conv!

* base on dcgans, cyclegan
* base on [mygans](https://github.com/JiangWeixian/GANS)

### Pre-works

* [x] - nets
    * [x] build dcgans
    * [x] init conv networks
    * [x] add layers's size to nets
* [ ] - training
    * [x] - add cuda to gans
* [x] - vision
    * [x] - add tensorflowboard support by crayon
* [x] - test_mnist
    * [x] - change g_loss_format(in ./test_mnist_dcgan.py)
* [ ] - test(pytorch/examples/dcgan.py)
    * [x] - download datasets-cifar10
    * [ ] - run this demo

