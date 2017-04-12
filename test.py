import numpy as np

def testparams(a, b, c, d=1):
    print a
    print b
    print c
    print d

def testzip():
    i = [1]*3
    jlist = [2, 3, 4]
    print zip(i, jlist)

def testtype():
    item = (1, 2)
    print type(item)

def testrandom():
    print np.random.rand(1)*0.1

if __name__ == '__main__':
    config = [1, 2, 3]
    testrandom()
    #testtype()
    #testzip()
    #testparams(*config)