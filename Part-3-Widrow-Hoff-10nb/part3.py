import sys
import argparse
import random
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import copy

sys.setrecursionlimit(1000000000)
load_dotenv()

LAYER_SIZES = [48, 5]
EPSILON = 0.1
THETA = 0.5
# FILE_TRAIN_LIST = ['0.txt', '1.txt', '2.txt']
# FILE_VERIF_LIST = ['0.txt', '1.txt', '2.txt']
# FILE_TEST_LIST = ['0.txt', '1.txt', '2.txt']
FILE_TRAIN_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt']
FILE_VERIF_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt']
FILE_TEST_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt']
# FILE_TRAIN_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt', '7.txt', '8.txt', '9.txt']
# FILE_VERIF_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt', '7.txt', '8.txt', '9.txt']
# FILE_TEST_LIST = ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt', '7.txt', '8.txt', '9.txt']
PATH = os.getenv("PATH_TO_FILE")

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
            if (char == '1' or char == '0'):
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
    # printImageTab(imageTab)
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

    # print(weightTab)
    return weightTab

# Randomize the choice of file to be used for training - use FILE_TRAIN_LIST list
def randFileChoice():
    return random.choice(FILE_TRAIN_LIST)

# Get the potential of the output neuron ((sum 1..j of Wij * Xj) - Theta)
def potentialOutputNeuronCalcul(weightTab, imageTab) :
    pot = []
    localResult = 0
    i = LAYER_SIZES[1] - 1


    for i in range(LAYER_SIZES[1]):
        for j in range(LAYER_SIZES[0]):
            localResult += weightTab[i][j] * imageTab[j]
        pot.append(localResult - THETA)

    # print("\npotential")
    # print(pot)
    return pot

# Learn phase -> training of the weight array
def learn(weightTab, error, imageTab) :
    for i in range(LAYER_SIZES[1]):
        for j in range(LAYER_SIZES[0]):
            weightTab[i][j] += (EPSILON * error[i] * imageTab[j])

    # print("weightTab")
    # print(weightTab)
    return weightTab

def verifNumber(fileName, weightTab, noise = 0) :
    loadFile = readFile(fileName)
    imageTab = loadFile.get('imageTab')
    if noise != 0:
        imageTab = sortingNoise(imageTab, noise)
    rightValueSaved = [0] * LAYER_SIZES[1]
    rightValueSaved[int(loadFile.get('value'))] = 1

    potOutput = potentialOutputNeuronCalcul(weightTab, imageTab)

    # Use heaviside to test it with generalization
    if noise > 0:
        # imageFound = -1
        # if (abs(potOutput) >= 0.5) :
        #     imageFound = 1
        # else :
        #     imageFound = 0
        # error = int(rightValueSaved) - imageFound
        # return error
        for i in range(LAYER_SIZES[1]):
            imageFound = -1
            error = [0] * LAYER_SIZES[1]
            if (abs(potOutput[i]) >= 0.5) :
                imageFound = 1
            else :
                imageFound = 0
            error[i] = abs(int(rightValueSaved[i]) - imageFound)
        # print(sum(error))
        return sum(error)

    error = [0] * LAYER_SIZES[1]
    for i in range(LAYER_SIZES[1]):
        error[i] = rightValueSaved[i] - potOutput[i]

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
        # print(imageNumber,' => ', 'True' if isNumberRecognized == 0 else 'False')
        cptImageErrorTab.append(abs(isNumberRecognized))
    return cptImageErrorTab

# Function which will train the model and validate or not after each iteration (stop method)
def trainNeuronNetwork(weightTab, cpt, errorTab) :
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
    if (verifError > 0.000000001):
        trainNeuronNetwork(newWeight, cpt+1, errorTab)
    else :
        print(cpt)
        # print(weightTab)
        # print(errorTab)
        plt.plot(errorTab)
        plt.show()
        # pass

    return newWeight

if __name__ == "__main__":
    trainedWeight = trainNeuronNetwork([], 0, [])
    # At the end we check with verif file (others files not in the training part)
    # print('\nVerification PART')
    testTrainedModel(trainedWeight, 0.3)
    # testTrainedModel(trainedWeight, 0.01)
    # testTrainedModel(trainedWeight, 0.01)
    # testTrainedModel(trainedWeight, 0.01)
