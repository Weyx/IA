import sys
import matplotlib.pyplot as plt
import gzip
import numpy as np
import time
import idx2numpy
from random import randrange

# imageFile = gzip.open('./samples/train-images-idx3-ubyte.gz','r')
# imageFile.read(16)
# labelFile = gzip.open('./samples/train-labels-idx1-ubyte.gz','r')
# labelFile.read(8)
# MAX_IMAGE_TRAIN = 59999

imageFile = gzip.open('./samples/t10k-images-idx3-ubyte.gz','r')
imageFile.read(16)
labelFile = gzip.open('./samples/t10k-labels-idx1-ubyte.gz','r')
labelFile.read(8)
MAX_IMAGE_TRAIN = 100

# LAYER_SIZES = [3,2,1]
IMAGE_SIZE = 28
LAYER_SIZES = [IMAGE_SIZE * IMAGE_SIZE, 100, 10]
EPSILON = 1
SHOW_IMG = 0

FILE = './samples/train-images-idx3-ubyte/train-images.idx3-ubyte'
LABEL = './samples/train-labels-idx1-ubyte/train-labels.idx1-ubyte'
ARR_FILES = idx2numpy.convert_from_file(FILE)
ARR_LABELS = idx2numpy.convert_from_file(LABEL)

def reloadImages():
    # global imageFile
    # imageFile = gzip.open('./samples/train-images-idx3-ubyte.gz','r')
    # imageFile.read(16)
    # global labelFile
    # labelFile = gzip.open('./samples/train-labels-idx1-ubyte.gz','r')
    # labelFile.read(8)

    global imageFile
    imageFile = gzip.open('./samples/t10k-images-idx3-ubyte.gz','r')
    imageFile.read(16)
    global labelFile
    labelFile = gzip.open('./samples/t10k-labels-idx1-ubyte.gz','r')
    labelFile.read(8)

# --- READ IMAGE INTO DATASET ---
def ascii_show(image):
    print("\n\n")
    for y in image:
        row = ""
        for x in y:
            row += '{0: <4}'.format(x)
        print(row)

def readNewImage1():
    index = randrange(1000)
    # ascii_show(ARR_FILES[index])
    # print(ARR_LABELS[index])

    imageConcatened = np.concatenate(ARR_FILES[index][0:IMAGE_SIZE])

    returnedValues = dict()
    returnedValues['imageTab'] = imageConcatened
    returnedValues['label'] = ARR_LABELS[index]
    # print(imageConcated)
    return returnedValues

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

    # global SHOW_IMG
    # if (SHOW_IMG < 5):
    #     ascii_show(image)
    #     SHOW_IMG +=1
    #     print(label)

    returnedValues = dict()
    returnedValues['imageTab'] = imageConcated
    returnedValues['label'] = label[0]

    # print(imageConcated)
    return returnedValues

# --- Init weight tab ---
def initWeightTab():
    weightTab1 = np.random.rand(LAYER_SIZES[1], LAYER_SIZES[0]) / (IMAGE_SIZE * IMAGE_SIZE)
    weightTab2 = np.random.rand(LAYER_SIZES[2], LAYER_SIZES[1]) / 100
    return [weightTab1, weightTab2]

# --- Calculate potential - PROPAGATION PART ---
def potOutputLayer1Calcul(weightL1, imageTab):
    # tic = time.perf_counter()
    # pot = []
    potH = []

    # localResult = 0
    # for h in range(LAYER_SIZES[1]):
    #     localResult = 0
    #     for j in range(LAYER_SIZES[0]):
    #         localResult += weightL1[h][j] * imageTab[j]
    #     pot.append(localResult)

    for h in np.arange(LAYER_SIZES[1]):
        potH.append(np.sum(weightL1[h] * imageTab))

    # toc = time.perf_counter()
    # print(f"potOutput1 function => {toc - tic:0.4f} seconds")
    # print(pot)
    # print(potH)s
    return potH

def Fx(x):
    return (1 / (1 + np.exp(-x)))

def functionAfterPot (potentialTab):
    for i in range(len(potentialTab)):
        potentialTab[i] = Fx(potentialTab[i])
    return potentialTab

def potOutputLayer2Calcul(weightL2, Xh):
    # tic = time.perf_counter()
    # pot = []
    potI = []

    # localResult = 0
    # for i in range(LAYER_SIZES[2]):
    #     localResult = 0
    #     for h in range(len(Xh)):
    #         localResult += weightL2[i][h] * Xh[h]
    #     pot.append(localResult)

    for i in np.arange(LAYER_SIZES[2]):
        potI.append(np.sum(weightL2[i] * Xh))

    # toc = time.perf_counter()
    # print(f"potOutput2 function => {toc - tic:0.4f} seconds")
    # print(pot)
    # print(potI)
    return potI

# --- Error calcul - RETROPROPAGATION PART ---
def calculateOutputLayerError (potI, xI, labelTab):
    sigmaI = [0] * LAYER_SIZES[2]

    for i in range(LAYER_SIZES[2]):
        fx = Fx(potI[i])
        derivate = fx * (1 - fx)
        sigmaI[i] = (derivate * labelTab[i]) - xI[i]

    # print(sigmaI)
    return sigmaI

def calculateHiddenLayerError(potH, sigmaI, weightL2):
    sigmaH = [0] * LAYER_SIZES[1]

    for h in range(LAYER_SIZES[1]):
        fx = Fx(potH[h])
        derivate = fx * (1 - fx)

        localSum = 0
        for i in range(len(sigmaI)):
            localSum += sigmaI[i] * weightL2[i][h]

        sigmaH[h] = derivate * localSum

    # print(sigmaH)
    return sigmaH

# --- LEARNING PART ---
def learning (sigmaI, sigmaH, weightL1, weightL2, Xj, Xh):
    # Wih (t+1) = Wih(t)+eps.sigma i . Xh
    # Whj (t+1) = Whj(t)+eps.sigma h . Xj
    # for i in range(len(sigmaI)):
    #     for h in range(len(Xh)):
    #         weightL2[i][h] += EPSILON * sigmaI[i] * Xh[h]

    XhSize = len(Xh)
    Xh = np.tile(Xh, (len(sigmaI),1))
    sigmaI = np.array(XhSize*[sigmaI])
    sigmaI = np.swapaxes(sigmaI, 0, 1)
    weightL2 += EPSILON *  Xh * sigmaI

    # TOO LONG ! => keep it -> easier to understand
    # Whj (t+1) = Whj(t)+eps.sigma h . Xj
    # for h in range(len(sigmaH)):
    #     for j in range(len(Xj)):
    #         # print(EPSILON * sigmaH[h] * Xj[j])
    #         weightL1[h][j] += EPSILON * sigmaH[h] * Xj[j]

    XjSize = len(Xj)
    Xj = np.tile(Xj, (len(sigmaH),1))
    sigmaH = np.array(XjSize*[sigmaH])
    sigmaH = np.swapaxes(sigmaH, 0, 1)
    weightL1 += EPSILON *  Xj * sigmaH

    # print(weightL1.size)
    # print(len(Xj))
    # print(weightL2)
    return [weightL1, weightL2];

def calculateErrorPercentageOn100Images (weightTab):
    cpt = 0
    sumAbsSigmaI = 0
    maxNumber = 100
    while cpt < maxNumber :
        returnedValue = readNewImage()
        # print(returnedValue.get("label"))
        # print(returnedValue.get("imageTab"))
        label = returnedValue.get("label")
        imageTab = returnedValue.get("imageTab") / 255

        # Indicate which label is it => example : if label = 8 => [0,0,0,0,0,0,0,0,1,0]
        labelTab = np.array([0] * LAYER_SIZES[2])
        labelTab[label] = 1

        # --- Propagation ---
        potentialOutputLayer1 = potOutputLayer1Calcul(weightTab[0], imageTab)
        Xh = functionAfterPot(potentialOutputLayer1)
        # print(len(potentialOutputLayer1))
        potentialOutputLayer2 = potOutputLayer2Calcul(weightTab[1], Xh)
        Xi = functionAfterPot(potentialOutputLayer2)
        # print(funcAfterPot2)

        # --- Retropropagation ---
        sigmaI = calculateOutputLayerError(potentialOutputLayer2, Xi, labelTab)

        sumAbsSigmaI += (np.sum(np.abs(sigmaI)))

        cpt +=1
    sumAbsSigmaI /= maxNumber
    # print(sumAbsSigmaI)
    return sumAbsSigmaI


def calculateErrorPercentage(sigmaI):
    return np.sum(np.abs(sigmaI))

def updateEpsilon (nb) :
    global EPSILON
    EPSILON = nb

def launchLearningPart(cpt, weightTab):
    errorTot = []
    errorLast100 = []
    timeAvr = 0
    cptLocal = 0
    epsilonUpdate = 0
    testData = []
    checkError = 0

    weightTab = initWeightTab()

    while cpt < 5000000000 :

        # tic = time.perf_counter()

        returnedValue = readNewImage1()
        # returnedValue = readNewImage()

        # print(returnedValue.get("label"))
        # print(returnedValue.get("imageTab"))
        label = returnedValue.get("label")
        imageTab = returnedValue.get("imageTab") / 255
        # print(imageTab)
        # print(len(imageTab))

        # Indicate which label is it => example : if label = 8 => [0,0,0,0,0,0,0,0,1,0]
        labelTab = np.array([0] * LAYER_SIZES[2])
        labelTab[label] = 1
        # print(labelTab)

        # --- Propagation ---
        potH = potOutputLayer1Calcul(weightTab[0], imageTab)
        Xh = functionAfterPot(potH)
        # print(len(potH))
        potI = potOutputLayer2Calcul(weightTab[1], Xh)
        Xi = functionAfterPot(potI)
        # print(funcAfterPot2)

        # --- Retropropagation ---
        sigmaI = calculateOutputLayerError(potI, Xi, labelTab)
        sigmaH = calculateHiddenLayerError(potH, sigmaI, weightTab[1])
        # print(hiddenLayerError)

        # --- Learning ---
        # tic = time.perf_counter()
        if (checkError == 0) :
            weightTab = learning(sigmaI, sigmaH, weightTab[0], weightTab[1], imageTab, Xh)
        else :
            errorLast100.append(np.sum(np.abs(sigmaI)))
            checkError += 1
            if (checkError == 100):
                totalSum = np.sum(errorLast100)/len(errorLast100)
                errorLast100.clear()
                if epsilonUpdate == 0 and totalSum <= 0.25:
                    updateEpsilon(0.1)
                    epsilonUpdate += 1
                if epsilonUpdate == 1 and totalSum <= 0.11:
                    updateEpsilon(0.01)
                    epsilonUpdate += 1
                if epsilonUpdate == 2 and totalSum <= 0.07:
                    updateEpsilon(0.001)
                    epsilonUpdate += 1

                print(cpt,EPSILON, " -> ",totalSum)
                checkError = 0

        if (cpt % 1000 == 0):
            checkError = 1

        # toc = time.perf_counter()
        # print(f"potOutput1 function => {toc - tic:0.4f} seconds")

        # --- Error ---
        errorTot.append(calculateErrorPercentage(sigmaI))

        # errorLast100.append(np.sum(np.abs(sigmaI)))
        # if (cpt % MAX_IMAGE_TRAIN == 0):
        #     testData.append(sigmaI[0])
        #     totalSum = np.sum(errorLast100)/len(errorLast100)
        #     # totalSum1 = np.sum(np.abs(sigmaI))
        #     # totalSum = calculateErrorPercentageOn100Images(weightTab)
        #     errorLast100.clear()
        #     if epsilonUpdate == 0 and totalSum <= 0.25:
        #         updateEpsilon(0.1)
        #         epsilonUpdate += 1
        #     if epsilonUpdate == 1 and totalSum <= 0.16:
        #         updateEpsilon(0.01)
        #         epsilonUpdate += 1
        #     if epsilonUpdate == 2 and totalSum <= 0.07:
        #         updateEpsilon(0.001)
        #         epsilonUpdate += 1

        #     print(cpt,EPSILON, " -> ",totalSum)

            # calculateErrorPercentageOn100Images(weightTab)

        # if (cptLocal >= MAX_IMAGE_TRAIN):
        #     print("RELOAD")
        #     reloadImages()
        #     cptLocal = 0
            # global SHOW_IMG
            # SHOW_IMG += 1
            # if (SHOW_IMG == 20):
            #     SHOW_IMG = 0
            #     plt.plot(testData, label="testData SigmaI[0]")
            #     plt.show()

        # cptLocal += 1
        cpt += 1
        # toc = time.perf_counter()
        # timeAvr += toc - tic
        # print(f"potOutput1 function => {toc - tic:0.4f} seconds")
    # print(timeAvr / 200)
    # calculateErrorPercentageOn100Images(weightTab)

    print(calculateErrorPercentage(sigmaI))

    bars=list(range(0, LAYER_SIZES[1]))
    plt.plot(errorTot, label="error")
    # plt.show()


if __name__ == "__main__":
    launchLearningPart(1, [])


#https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python => first url used to read in gz file and extract data
