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

imageFile = gzip.open('./samples/t10k-images-idx3-ubyte.gz','r')
imageFile.read(16)
labelFile = gzip.open('./samples/t10k-labels-idx1-ubyte.gz','r')
labelFile.read(8)

# LAYER_SIZES = [3,2,1]
IMAGE_SIZE = 28
LAYER_SIZES = [IMAGE_SIZE * IMAGE_SIZE, 100, 10]
EPSILON = 1
SHOW_IMG = 0

NB_IMG_TRAIN = 60000
NB_IMG_TEST = 10000
RATES = [0.25, 0.07, 0.04]
RATE_ERROR_MIN = 0.09
TEST_NB = 10000
MAX_ITERATION_TRAIN = 300000

FILES_TRAIN = './samples/train-images-idx3-ubyte/train-images.idx3-ubyte'
LABELS_TRAIN = './samples/train-labels-idx1-ubyte/train-labels.idx1-ubyte'
ARR_FILES_TRAIN = idx2numpy.convert_from_file(FILES_TRAIN)
ARR_LABELS_TRAIN = idx2numpy.convert_from_file(LABELS_TRAIN)

FILES_TEST = './samples/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte'
LABELS_TEST = './samples/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte'
ARR_FILES_TEST = idx2numpy.convert_from_file(FILES_TEST)
ARR_LABELS_TEST = idx2numpy.convert_from_file(LABELS_TEST)



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

def readNewImage1(choice):
    returnedValues = dict()

    if (choice == "train"):
        index = randrange(NB_IMG_TRAIN)
        # ascii_show(ARR_FILES_TRAIN[index])
        # print(ARR_LABELS_TRAIN[index])
        imageConcatened = np.concatenate(ARR_FILES_TRAIN[index][0:IMAGE_SIZE])
        returnedValues['label'] = ARR_LABELS_TRAIN[index]
    else :
        index = randrange(NB_IMG_TEST)
        imageConcatened = np.concatenate(ARR_FILES_TEST[index][0:IMAGE_SIZE])
        returnedValues['label'] = ARR_LABELS_TEST[index]
        # ascii_show(ARR_FILES_TEST[index])
        # print(ARR_LABELS_TEST[index])


    returnedValues['imageTab'] = imageConcatened
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

    # CODE OPTIMIZATION
    # Time1 => 0.07s / image (with 2 for loops)
    # Time2 => 0.0056s / image (with the second way to update weights)
    # Time3 => 0.0006s / image (with transpose method)


    # for i in range(len(sigmaI)):
    #     for h in range(len(Xh)):
    #         weightL2[i][h] += EPSILON * sigmaI[i] * Xh[h]

    # XhSize = len(Xh)
    # Xh = np.tile(Xh, (len(sigmaI),1))
    # sigmaI = np.array(XhSize*[sigmaI])
    # sigmaI = np.swapaxes(sigmaI, 0, 1)
    # weightL2 += EPSILON *  Xh * sigmaI

    Xh = np.tile(Xh, (len(sigmaI),1))
    weightL2 += EPSILON *  Xh * np.transpose(np.array([sigmaI,]))

    # TOO LONG ! => keep it -> easier to understand
    # Whj (t+1) = Whj(t)+eps.sigma h . Xj
    # for h in range(len(sigmaH)):
    #     for j in range(len(Xj)):
    #         # print(EPSILON * sigmaH[h] * Xj[j])
    #         weightL1[h][j] += EPSILON * sigmaH[h] * Xj[j]

    # XjSize = len(Xj)
    # Xj = np.tile(Xj, (len(sigmaH),1))
    # sigmaH = np.array(XjSize*[sigmaH])
    # sigmaH = np.swapaxes(sigmaH, 0, 1)
    # weightL1 += EPSILON * Xj * sigmaH

    Xj = np.tile(Xj, (len(sigmaH),1))
    weightL1 += EPSILON * Xj * np.transpose(np.array([sigmaH,]))

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

def modelTested(weightTab):
    cpt = 0
    error = 0

    while cpt < TEST_NB :
        returnedValue = readNewImage1("test")
        label = returnedValue.get("label")
        imageTab = returnedValue.get("imageTab") / 255.0

        # --- Propagation ---
        potH = potOutputLayer1Calcul(weightTab[0], imageTab)
        Xh = functionAfterPot(potH)
        potI = potOutputLayer2Calcul(weightTab[1], Xh)
        Xi = functionAfterPot(potI)

        # print(label)
        # print(Xi, "\n")

        valueFound = np.argmax(Xi)
        # print("Value found is ", valueFound, "(exact value : ",label,")")
        # print(sigmaI, "\n")

        if (valueFound != label):
            error += 1

        cpt += 1
    print("Nb error : ", error)

def launchLearningPart(cpt, weightTab):
    errorLast100 = []
    timeAvr = 0
    epsilonUpdate = 0
    checkError = 0
    sigmaI_SAVED = {}
    totalSum = 1

    weightTab = initWeightTab()

    # while cpt < 500000000000 :
    while totalSum > RATE_ERROR_MIN and cpt < MAX_ITERATION_TRAIN :

        # tic = time.perf_counter()
        # toc = time.perf_counter()
        # print(f"1 image => {toc - tic:0.4f} seconds")

        returnedValue = readNewImage1("train")
        # returnedValue = readNewImage()

        # print(returnedValue.get("label"))
        # print(returnedValue.get("imageTab"))

        label = returnedValue.get("label")
        imageTab = returnedValue.get("imageTab") / 255.0
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
            # if (checkError == 99):
                # print('\n')
                # print(sigmaI)
                # print(np.abs(sigmaI))
                # print(np.sum(np.abs(sigmaI)))
                # print(np.sort(errorLast100))
                # print(np.sort(sigmaI_SAVED))
                # for i in sorted (sigmaI_SAVED) :
                #     print ((i, sigmaI_SAVED[i]), end =" ")
                # print('\n')
            sumSigmaI = np.sum(np.abs(sigmaI))
            sigmaI_SAVED[sumSigmaI] = label
            errorLast100.append(sumSigmaI)
            checkError += 1
            if (checkError == 100):
                totalSum = np.sum(errorLast100)/len(errorLast100)
                errorLast100.clear()
                sigmaI_SAVED = {}
                if epsilonUpdate == 0 and totalSum <= RATES[0]:
                    updateEpsilon(0.1)
                    epsilonUpdate += 1
                if epsilonUpdate == 1 and totalSum <= RATES[1]:
                    updateEpsilon(0.01)
                    epsilonUpdate += 1
                if epsilonUpdate == 2 and totalSum <= RATES[2]:
                    updateEpsilon(0.001)
                    epsilonUpdate += 1

                print(cpt,EPSILON, NB_IMG_TRAIN, " -> ",totalSum, RATES, RATE_ERROR_MIN, TEST_NB, MAX_ITERATION_TRAIN)
                checkError = 0

        if (cpt % 1000 == 0):
            checkError = 1

        # toc = time.perf_counter()
        # print(f"1 image => {toc - tic:0.4f} seconds")

        # --- Error ---
        # calculateErrorPercentage(sigmaI)

        cpt += 1
        # toc = time.perf_counter()
        # timeAvr += toc - tic
        # print(f"potOutput1 function => {toc - tic:0.4f} seconds")
    # print("temps moyen ", timeAvr / 200)
    # print(timeAvr)
    # calculateErrorPercentageOn100Images(weightTab)

    # bars=list(range(0, LAYER_SIZES[1]))
    # plt.plot(errorTot, label="error")
    # plt.show()

    return weightTab


if __name__ == "__main__":
    weightTab = launchLearningPart(1, [])
    modelTested(weightTab)


#https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python => first url used to read in gz file and extract data
