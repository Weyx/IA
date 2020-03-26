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
    # Wih (t+1) = Wih(t)+eps.sigma i . Xh
    weightL2 = np.arange(6, dtype=float).reshape(2,3)
    sigmaI = np.array([0.3, 0.6])
    Xh = np.array([0,1,0])
    XhSize = len(Xh)

    print(sigmaI)
    print(Xh)
    Xh = np.tile(Xh, (len(sigmaI),1))
    sigmaI = np.array(XhSize*[sigmaI])
    sigmaI = np.swapaxes(sigmaI, 0, 1)
    weightL2 += EPSILON *  Xh * sigmaI
    print(weightL2)

# OK
def testPotH():
    imageTab = np.array([0., 0., 0., 0.2, 0.5, 0.])
    weightL1 = np.arange(12, dtype=float).reshape(2,6)
    print(weightL1)

    potH = []
    # localResult = 0
    # for h in range(2):
    #     localResult = 0
    #     for j in range(6):
    #         localResult += weightL1[h][j] * imageTab[j]
    #     potH.append(localResult)

    for h in np.arange(2):
        potH.append(np.sum(weightL1[h] * imageTab))

    print(potH)
    return potH

def Fx(x):
    return (1 / (1 + np.exp(-x)))

# OK
def functionAfterPot (potentialTab):
    for i in range(len(potentialTab)):
        potentialTab[i] = Fx(potentialTab[i])
    return potentialTab

# OK
def calculateOutputLayerError ():
    potI = np.array([1,2])
    xI = np.array([0.2, 0.5])
    labelTab = np.array([0, 1])

    sigmaI = [0] * 2
    for i in range(2):
        fx = Fx(potI[i])
        derivate = fx * (1 - fx)
        sigmaI[i] = (derivate * labelTab[i]) - xI[i]

    print(sigmaI)
    return sigmaI

# OK
def calculateHiddenLayerError():
    sigmaH = [0] * 3
    potH = np.arange(3)
    sigmaI = np.arange(2)
    weightL2 = np.arange(6, dtype=float).reshape(2,3)

    print(sigmaI)
    print(weightL2)

    for h in range(3):
        fx = Fx(potH[h])
        derivate = fx * (1 - fx)
        print(derivate)

        localSum = 0
        for i in range(2):
            localSum += sigmaI[i] * weightL2[i][h]
            print(localSum)

        sigmaH[h] = derivate * localSum

    print(sigmaH)
    return sigmaH

if __name__ == "__main__":

    # goodWork()
    # testPotH()
    # potH = testPotH()
    # print(functionAfterPot(potH))
    # calculateOutputLayerError()
    # calculateHiddenLayerError()

    # a = np.random.rand(2,3)
    # b = np.fromfunction(test, a.shape)
    # print(a)
    # # b = np.fromfunction(test,(5,4),dtype=int)
    # print(b)

    # sigmaI = [-2, 3, 0.5]
    # print(np.sum(np.abs(sigmaI)))

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
