import sys
import argparse
import random
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import copy
import numpy as np

load_dotenv()

EPSILON = 0.01
THETA = 0.5

VERIF_ERROR = 0.000001
# FILE_TRAIN_LIST = ['0.txt', '1.txt']
# FILE_VERIF_LIST = ['0.txt', '1.txt']
# FILE_TEST_LIST = ['0.txt', '1.txt']
# FILE_TRAIN_LIST = ['0.txt', '1.txt', '2.txt']
# FILE_VERIF_LIST = ['0.txt', '1.txt', '2.txt']
# FILE_TEST_LIST = ['0.txt', '1.txt', '2.txt']
# FILE_TRAIN_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt']
# FILE_VERIF_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt']
# FILE_TEST_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt']
# FILE_TRAIN_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt']
# FILE_VERIF_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt']
# FILE_TEST_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt']
# FILE_TRAIN_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt']
# FILE_VERIF_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt']
# FILE_TEST_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt'] #12k
# FILE_TRAIN_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt', '7.txt']
# FILE_VERIF_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt', '7.txt']
# FILE_TEST_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt', '7.txt'] #15k
# FILE_TRAIN_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt', '7.txt', '8.txt']
# FILE_VERIF_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt', '7.txt', '8.txt']
# FILE_TEST_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt', '7.txt', '8.txt'] #23k
FILE_TRAIN_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt', '7.txt', '8.txt', '9.txt']
FILE_VERIF_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt', '7.txt', '8.txt', '9.txt']
FILE_TEST_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt', '7.txt', '8.txt', '9.txt'] #?k

PATH = os.getenv("PATH_TO_FILE")
LAYER_SIZES = [48, len(FILE_TRAIN_LIST)]

imageCached = {}

def printImageTab(imageTab):
    for i in range(len(imageTab)):
        print ('*' if imageTab[i] == 1 else '.', end="") if (i+1)%6 != 0 else print('*' if imageTab[i] == 1 else '.')

def readFile(choice):
    # Try to cache image - generate bug into returned errors
    if(imageCached.get(choice) == None):
        imageTab = []
        value = -1
        f= open(PATH+choice,"r")
        # Read file char by char
        while True:
            char = f.read(1)
            if not char:
                # print "End of file"
                break
            if (char == '*'):
                imageTab.append(1)
            if (char == '.'):
                imageTab.append(0)
            if (char == '0' or
                char == '1' or
                char == '2' or
                char == '3' or
                char == '4' or
                char == '5' or
                char == '6' or
                char == '7' or
                char == '8' or
                char == '9'):
                value = char

        dictionnary = dict()
        dictionnary['imageTab'] = imageTab
        dictionnary['value'] = value
        imageCached[choice] = dictionnary
        return dictionnary
    else :
        # Return a copy of the cache
        return copy.deepcopy(imageCached.get(choice))

def sortingNoise (imageTab, percentage = 0) :
    reverseNumber = int(LAYER_SIZES[0] * percentage)
    if (reverseNumber <= LAYER_SIZES[0]):
        reverseIndexTab = random.sample(range(0, LAYER_SIZES[0]), reverseNumber)
    for item in reverseIndexTab:
        imageTab[item] = 1 if (imageTab[item] == 0) else 0
    return imageTab

def initWeightTab():
    weightTab = []
    # Init tab
    for i in range(LAYER_SIZES[1]):
        weightTab.append([0] * LAYER_SIZES[0])

    for i in range(LAYER_SIZES[1]):
        for j in range(LAYER_SIZES[0]):
            rnd = random.random()
            weightTab[i][j] = rnd  / 48

    return weightTab

# Randomize the choice of file to be used for training - use FILE_TRAIN_LIST list
def randFileChoice():
    return random.choice(FILE_TRAIN_LIST)

# Get the potential of the output neuron ((sum 1..j of Wij * Xj) - Theta)
def potentialOutputNeuronCalcul(weightTab, imageTab) :
    pot = []
    localResult = 0

    for i in range(LAYER_SIZES[1]):
        localResult = 0
        for j in range(LAYER_SIZES[0]):
            localResult += weightTab[i][j] * imageTab[j]
        pot.append(localResult - THETA)

    return pot

# Learn phase -> training of the weight array
def learn(weightTab, error, imageTab) :
    for i in range(LAYER_SIZES[1]):
        for j in range(LAYER_SIZES[0]):
            weightTab[i][j] += (EPSILON * error[i] * imageTab[j])

    return weightTab

def verifNumber(fileName, weightTab, noise = 0) :
    loadFile = readFile(fileName)
    imageTab = loadFile.get('imageTab')
    if noise != 0:
        imageTab = sortingNoise(imageTab, noise)
    rightValueSaved = [0] * LAYER_SIZES[1]
    rightValueSaved[int(loadFile.get('value'))] = 1

    # print(loadFile.get('value'))

    potOutput = potentialOutputNeuronCalcul(weightTab, imageTab)

    error = [0] * LAYER_SIZES[1]
    for i in range(LAYER_SIZES[1]):
        error[i] = rightValueSaved[i] - potOutput[i]

    # Only called during tests
    if noise > 0 :
        # print(rightValueSaved)
        # print(potOutput)
        return [potOutput, rightValueSaved]

        # Print histogram for each number
        # bars=list(range(0, LAYER_SIZES[1]))
        # plt.bar(bars, potOutput)
        # plt.show()

    return sum(error)

def verifPart(weightTab, noise = 0):
    resultFinalError = 0
    errorList = []
    for imageNumber in FILE_VERIF_LIST:
        checkNumberIsRecognized = verifNumber(imageNumber, weightTab)
        resultFinalError += abs(checkNumberIsRecognized)
        errorList.append(checkNumberIsRecognized)

    return resultFinalError

def testTrainedModel(weightTabTrained, noise = 0):
    cptImageErrorTab = []
    for imageNumber in FILE_TEST_LIST:
        isNumberRecognized = verifNumber(imageNumber, weightTabTrained, noise)
        cptImageErrorTab.append(abs(isNumberRecognized))
    return cptImageErrorTab

def testTrainedModel100(weightTabTrained, noise = 0):
    cptNb = [0] * LAYER_SIZES[1]
    nbTry = 1000
    for imageNumber in FILE_TEST_LIST:
        cptNb = [0] * LAYER_SIZES[1]
        for i in range(nbTry) :
            potOutput = verifNumber(imageNumber, weightTabTrained, noise)
            result = np.where(potOutput[0] == np.amax(potOutput[0]))
            cptNb[result[0][0]] += 1
            # print(result[0])
            # print("\n")
        print(cptNb)
        rightNumber = result = np.where(potOutput[1] == np.amax(potOutput[1]))[0][0]
        # Print histogram for each number
        bars=list(range(0, LAYER_SIZES[1]))
        plt.bar(bars, cptNb)
        plt.title("Nombre : "+str(rightNumber)+" (Bruitage: "+str(noise)+" - Essais: "+str(nbTry)+")")
        plt.savefig('generatedPlots/Noise/Noise_'+str(noise)+'/Nb'+str(rightNumber)+'_Noise'+str(noise)+'.png')
        plt.show()


# Function which will train the model and validate or not after each iteration (stop method)
def trainNeuronNetwork(weightTab, cpt, errorTab) :
    verifError = 1
    while verifError > VERIF_ERROR:
        # print(verifError)
        choiceFile = randFileChoice()
        loadFile = readFile(choiceFile)
        tab = loadFile.get('imageTab')
        rightValueSaved = [0] * LAYER_SIZES[1]
        rightValueSaved[int(loadFile.get('value'))] = 1

        # Init weight tab only at the beginning when the model is not yet trained
        if (len(weightTab) == 0):
            weightTab = initWeightTab()

        potOutput = potentialOutputNeuronCalcul(weightTab, tab)

        # 5 Error calcul
        error = [0] * LAYER_SIZES[1]
        for i in range(LAYER_SIZES[1]):
            error[i] = rightValueSaved[i] - potOutput[i]

        # 6 Learn
        newWeight = learn(weightTab, error, tab)

        # Train while models is still making errors
        verifError = verifPart(newWeight)
        errorTab.append(verifError)
        cpt +=1

    print(cpt)
    # print(weightTab)
    # print(errorTab)
    plt.plot(errorTab)
    plt.show()

    return newWeight

def printContrastedWeight(weightTab):
    test = []
    for i in range(8):
        test.append((weightTab[i * 6 + 0],
            weightTab[i * 6 + 1],
            weightTab[i * 6 + 2],
            weightTab[i * 6 + 3],
            weightTab[i * 6 + 4],
            weightTab[i * 6 + 5]))

    plt.imshow(test)
    # plt.imshow([(3,3,0),(0,2,0),(0,0,1)])
    plt.colorbar()
    plt.show()

def saveTrainedWeight (weightTab):
    with open("./saveWeights/trainedWeights.txt", "w") as txt_file:
        for line in weightTab:
            txt_file.write("".join(str(line)) + "\n")

if __name__ == "__main__":
    test = []
    trainedWeight = trainNeuronNetwork([], 0, [])

    # READ into file (saved trained weights)

    # At the end we check with verif file (others files not in the training part)
    print('\nVerification PART')
    # print(testTrainedModel(trainedWeight, 0))
    # testTrainedModel(trainedWeight, 0.7)
    testTrainedModel100(trainedWeight, 0.1)

    # print(testTrainedModel(trainedWeight, 0.1))
    # print(testTrainedModel(trainedWeight, 0.2))
    # print(testTrainedModel(trainedWeight, 0.3))
    # print(testTrainedModel(trainedWeight, 0.5))
    # testTrainedModel(trainedWeight, 0.01)
    # testTrainedModel(trainedWeight, 0.01)
    # testTrainedModel(trainedWeight, 0.01)

    # printContrastedWeight(trainedWeight[0])
    # printContrastedWeight(trainedWeight[1])
    # printContrastedWeight(trainedWeight[2])
    # printContrastedWeight(trainedWeight[3])
    # printContrastedWeight(trainedWeight[4])
    # printContrastedWeight(trainedWeight[5])
    # printContrastedWeight(trainedWeight[6])
    # printContrastedWeight(trainedWeight[7])
    # printContrastedWeight(trainedWeight[8])

    # print(trainedWeight)
    # saveTrainedWeight(trainedWeight)
