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

if __name__ == '__main__':
    config = [1, 2, 3]
    testtype()
    #testzip()
    #testparams(*config)