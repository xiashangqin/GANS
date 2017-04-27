import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def testparams(a, b, c, d=1):
    print a
    print b
    print c
    print d

def testzip():
    i = [1]*3
    jlist = [2, 3, 4]
    print zip(i, jlist)

def testzip2():
    i = [1]*3
    jlist = [2, 3, 4]
    ziplist = [ (iindex, jindex) for iindex, jindex in zip(i, jlist) ]
    print ziplist

def testtype():
    item = (1, 2)
    print type(item)

def testrandom():
    print np.random.rand(1)*0.1

def testadd():
    a = torch.randn(4)
    b = torch.randn(4)
    z = torch.zeros(1)
    print z
    print (a + b)/2
    # testsum
    elist = []
    elist.append(a)
    elist.append(b)
    b = sum(elist)
    print b

def testremovelist():
    a = [1, 2, 3, 4]
    length = range(len(a))
    extra = list(length).remove(2)
    print length

def testdir():
    print dir(list)

def testcontinue():
    a = [1, 2, 3, 4]
    for i in a:
        if i == 2:
            continue
        print i

def testyeild():
    a = [1, 2, 3]
    for i in a:
        yield i

def testzip3():
    result = testyeild()
    a = [1, 2, 3]
    print zip(result, a)

def list2npmax():
    a = np.array([1, 2, 3])
    print type(a)

def listresize():
    a = [1, 2, 3, 4]
    return a[1:3]

def testtuple():
    a = (1, 2)
    print type(a) == tuple

def testbackward():
    x = Variable(torch.ones(2, 2), requires_grad=True)
    y = x + 2
    z = Variable(torch.ones(2, 2), requires_grad=True)
    f = y * z 
    out = f.mean()
    print 'first\n'
    print x.grad
    out.backward()
    print 'second\n'
    print x.grad
    print 'second z\n'
    print z.grad
def testrange():
    for i in range(1):
        print i
def testbce():
    real = torch.ones(4, 1) * 0.2
    fake = torch.ones(4, 1) * 0.2
    real_target = torch.ones(4, 1)
    fake_target = torch.zeros(4, 1)
    real = Variable(real)
    fake = Variable(fake)
    real_target = Variable(real_target)
    fake_target = Variable(fake_target)
    criterion = nn.BCELoss()
    real_bce_loss = criterion(real, real_target)
    fake_bce_loss =  criterion(fake, fake_target)
    print fake_bce_loss
    real_loss = torch.mean(torch.log(real))
    fake_loss = torch.mean(torch.log(1 - fake))
    print fake_loss


if __name__ == '__main__':
    config = [1, 2, 3]
    testbce()
    #list2npmax()
    #testzip3()
    #result = testyeild()
    #testcontinue()
    #testzip2()
    #testdir()
    #testremovelist()
    #testadd()
    #testrandom()
    #testtype()
    #testzip()
    #testparams(*config)