import sys
import matplotlib.pyplot as plt
import gzip
import numpy as np
import time

EPSILON = 0.01

def test(x,y):
    print('test')
    return 10*x

def goodWork():
    weightL2 = np.arange(6, dtype=float).reshape(2,3)
    sigmaI = np.array([0.3, 0.6])
    Xh = np.array([0,1,0])
    XhSize = len(Xh)

    Xh = np.tile(Xh, (len(sigmaI),1))
    sigmaI = np.array(XhSize*[sigmaI])
    sigmaI = np.swapaxes(sigmaI, 0, 1)
    weightL2 += EPSILON *  Xh * sigmaI
    print(weightL2)

if __name__ == "__main__":
    goodWork()
    # a = np.random.rand(2,3)
    # b = np.fromfunction(test, a.shape)
    # print(a)
    # # b = np.fromfunction(test,(5,4),dtype=int)
    # print(b)

    # a = np.linspace(-.5, .5, 1001)
    # S = np.zeros((a.size, a.size))
    # S = np.sum((east*a[:, None, None] + north*a[:, None] - tp)**2, axis=-1)
    # print(a)
    # print(S)
    # print(a)
    # print(a[:,2])
    # print("\nLOOOL")

    # w = np.arange(6).reshape(2,3)
    weightL2 = np.arange(6, dtype=float).reshape(2,3)
    sigmaI = np.array([0.3, 0.6])
    Xh = [0,1,0]
    # print(weightL2)

    # for i in range(len(sigmaI)):
    #     for h in range(len(Xh)):
    #         weightL2[i][h] += EPSILON * sigmaI[i] * Xh[h]
    # print(weightL2)

    # print(Xh)
    # print(np.tile(Xh, (len(sigmaI),1)))
    # print(np.tile(sigmaI, (len(Xh), 1)))
    # print(np.array(3*[sigmaI]))
    Xh = np.tile(Xh, (len(sigmaI),1))
    sigmaI = np.array(3*[sigmaI])
    sigmaI = np.swapaxes(sigmaI, 0, 1)
    weightL2 += EPSILON *  Xh * sigmaI
    # print(weightL2)


    # print(a[:,np.newaxis])


#https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python => first url used to read in gz file and extract data
