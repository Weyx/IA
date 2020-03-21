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
    weightTab2 = np.random.rand(LAYER_SIZES[2], LAYER_SIZES[1]) / (IMAGE_SIZE * IMAGE_SIZE)


    # print("nombre de dimensions de x1: ", weightTab.ndim)
    # print("forme de weightTab: ", weightTab.shape)
    # print("taille de weightTab: ", weightTab.size)
    # print("type de weightTab: ", weightTab.dtype)

    return [weightTab1, weightTab2]

# --- Calculate potential ---
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
    # print(pot1)
    return pot1

def functionAfterPot (potentialTab):
    for i in range(len(potentialTab)):
        potentialTab[i] = 1 / (1 + np.exp(-potentialTab[i]))
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


if __name__ == "__main__":
    for i in range(1):
        returnedValue = readNewImage()
        # print(returnedValue.get("label"))
        # print(returnedValue.get("imageTab"))
        label = returnedValue.get("label")
        imageTab = returnedValue.get("imageTab") / 255
        # print(imageTab)

        weightTab = initWeightTab()
        # print(weightTab[1].size)

        potentialOutputLayer1 = potOutputLayer1Calcul(weightTab[0], imageTab)
        funcAfterPot1 = functionAfterPot(potentialOutputLayer1)
        # print(len(potentialOutputLayer1))
        potentialOutputLayer2 = potOutputLayer2Calcul(weightTab[1], funcAfterPot1)
        funcAfterPot2 = functionAfterPot(potentialOutputLayer2)
        # print(funcAfterPot2)

#https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python => first url used to read in gz file and extract data
