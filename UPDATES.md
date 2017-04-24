# updates for mygans
> gans更新记录

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