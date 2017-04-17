import numpy as np
import torch

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

if __name__ == '__main__':
    config = [1, 2, 3]
    testzip3()
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