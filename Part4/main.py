import sys
import matplotlib.pyplot as plt
import gzip
import numpy as np
import time

imageFile = gzip.open('./samples/train-images-idx3-ubyte.gz','r')
imageFile.read(16)
labelFile = gzip.open('./samples/train-labels-idx1-ubyte.gz','r')
labelFile.read(8)

# LAYER_SIZES = [3,2,1]
IMAGE_SIZE = 28
LAYER_SIZES = [IMAGE_SIZE * IMAGE_SIZE, 100, 10]
EPSILON = 0.01

# --- READ IMAGE INTO DATASET ---
def ascii_show(image):
    print("\n\n")
    for y in image:
        row = ""
        for x in y:
            row += '{0: <4}'.format(x)
        print(row)

def readNewImage():
    # READ IMAGE (28x28)
    nbImages = 1

    buf = imageFile.read(IMAGE_SIZE * IMAGE_SIZE * nbImages)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(nbImages, IMAGE_SIZE, IMAGE_SIZE, 1)

    image = np.asarray(data[0]).squeeze()
    imageConcated = np.concatenate(image[0:IMAGE_SIZE])
    # ascii_show(image)

    # READ LABELS (1, 2, 3, ..., 9)
    buf = labelFile.read(1)
    label = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    # print(label)

    returnedValues = dict()
    returnedValues['imageTab'] = imageConcated
    returnedValues['label'] = label[0]

    return returnedValues

# ---  ---
def initWeightTab():
    weightTab1 = np.random.rand(LAYER_SIZES[1], LAYER_SIZES[0]) / (IMAGE_SIZE * IMAGE_SIZE)
    weightTab2 = np.random.rand(LAYER_SIZES[2], LAYER_SIZES[1]) / (100)


    # print("nombre de dimensions de x1: ", weightTab.ndim)
    # print("forme de weightTab: ", weightTab.shape)
    # print("taille de weightTab: ", weightTab.size)
    # print("type de weightTab: ", weightTab.dtype)

    return [weightTab1, weightTab2]

# --- Calculate potential - PROPAGATION PART ---
def potOutputLayer1Calcul(weightL1, imageTab):
    print(weightL1.size)
    print(len(weightL1))
    tic = time.perf_counter()
    # pot = []
    pot1 = []

    # localResult = 0
    # for h in range(LAYER_SIZES[1]):
    #     localResult = 0
    #     for j in range(LAYER_SIZES[0]):
    #         localResult += weightL1[h][j] * imageTab[j]
    #     pot.append(localResult)

    for h in np.arange(LAYER_SIZES[1]):
        pot1.append(np.sum(weightL1[h] * imageTab))

    toc = time.perf_counter()
    print(f"potOutput1 function => {toc - tic:0.4f} seconds")
    # print(pot)
    # print(pot1)s
    return pot1

def Fx(x):
    return (1 / (1 + np.exp(-x)))

def functionAfterPot (potentialTab):
    for i in range(len(potentialTab)):
        potentialTab[i] = Fx(potentialTab[i])
    return potentialTab

def potOutputLayer2Calcul(weightL2, funcAfterPot1):
    print(weightL2.size)
    print(len(weightL2))
    tic = time.perf_counter()
    # pot = []
    pot2 = []

    # localResult = 0
    # for i in range(LAYER_SIZES[2]):
    #     localResult = 0
    #     for h in range(len(funcAfterPot1)):
    #         localResult += weightL2[i][h] * funcAfterPot1[h]
    #     pot.append(localResult)

    for i in np.arange(LAYER_SIZES[2]):
        pot2.append(np.sum(weightL2[i] * funcAfterPot1))

    toc = time.perf_counter()
    print(f"potOutput2 function => {toc - tic:0.4f} seconds")
    # print(pot)
    # print(pot2)
    return pot2

# --- Error calcul - RETROPROPAGATION PART ---
def calculateOutputLayerError (potTabOutput, funcAfterPotOutput, labelTab):
    errorOutputTab = [0] * LAYER_SIZES[2]

    for i in range(LAYER_SIZES[2]):
        fx = Fx(potTabOutput[i])
        derivate = fx * (1 - fx)
        errorOutputTab[i] = derivate * labelTab[i] - funcAfterPotOutput[i]

    # print(errorOutputTab)
    return errorOutputTab

def calculateHiddenLayerError(potTabHiddenLayer, errorOutputTab, weightL2):
    tic = time.perf_counter()
    errorHiddenLayerTab = [0] * LAYER_SIZES[1]

    for h in range(LAYER_SIZES[1]):
        fx = Fx(potTabHiddenLayer[h])
        derivate = fx * (1 - fx)
        # partWeightTab = weightL2[h]
        # sumErrorWeights = np.sum(errorOutputTab * partWeightTab)

        localSum = 0
        for i in range(len(errorOutputTab)):
            localSum += errorOutputTab[i] * weightL2[i][h]

        errorHiddenLayerTab[h] = derivate * localSum

    toc = time.perf_counter()
    print(f"calculateHiddenLayerError function => {toc - tic:0.4f} seconds")
    # print(errorHiddenLayerTab)
    return errorHiddenLayerTab

# --- LEARNING PART ---
def learning (sigmaI, sigmaH, weightL1, weightL2, Xj, Xh):
    # Wih (t+1) = Wih(t)+eps.sigma i . Xh
    # Whj (t+1) = Whj(t)+eps.sigma h . Xj
    for i in range(len(sigmaI)):
        for h in range(len(Xh)):
            weightL2[i][h] += EPSILON * sigmaI[i] * Xh[h]

    for h in range(len(sigmaH)):
        for j in range(len(Xj)):
            # print(EPSILON * sigmaH[h] * Xj[j])
            weightL1[i][h] += EPSILON * sigmaH[h] * Xj[j]

    # print(weightL1.size)
    # print(len(Xj))
    # print(weightL2)
    return ;

if __name__ == "__main__":
    for i in range(1):
        returnedValue = readNewImage()
        # print(returnedValue.get("label"))
        # print(returnedValue.get("imageTab"))
        label = returnedValue.get("label")
        imageTab = returnedValue.get("imageTab") / 255

        # Indicate which label is it => example : if label = 8 => [0,0,0,0,0,0,0,0,1,0]
        labelTab = np.array([0] * LAYER_SIZES[2])
        labelTab[label] = 1

        weightTab = initWeightTab()
        # print(weightTab[1].size)

        # --- Propagation ---
        potentialOutputLayer1 = potOutputLayer1Calcul(weightTab[0], imageTab)
        Xh = functionAfterPot(potentialOutputLayer1)
        # print(len(potentialOutputLayer1))
        potentialOutputLayer2 = potOutputLayer2Calcul(weightTab[1], Xh)
        Xi = functionAfterPot(potentialOutputLayer2)
        # print(funcAfterPot2)

        # --- Retropropagation ---
        sigmaI = calculateOutputLayerError(potentialOutputLayer2, Xi, labelTab)
        sigmaH = calculateHiddenLayerError(potentialOutputLayer1, sigmaI, weightTab[1])
        # print(hiddenLayerError)

        # --- Learning ---
        learning(sigmaI, sigmaH, weightTab[0], weightTab[1], imageTab, Xh)


#https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python => first url used to read in gz file and extract data
