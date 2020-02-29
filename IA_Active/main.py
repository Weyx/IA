import sys
import argparse
import random
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

LAYER_SIZES = [48, 1]
EPSILON = 0.01
THETA = 0.5
FILE_TRAIN_LIST = ['zero.txt', 'zero1.txt', 'one.txt', 'one1.txt']
FILE_VERIF_LIST = ['zero.txt', 'zero1.txt', 'one.txt', 'one1.txt']
FILE_TEST_LIST = ['one2.txt','zero2.txt']
PATH = os.getenv("PATH_TO_FILE")

def readFile(choice):
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
    return dictionnary

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
    pot = 0
    i = LAYER_SIZES[1] - 1

    for j in range(LAYER_SIZES[0]):
        pot += weightTab[i][j] * imageTab[j]

    return (pot - THETA)

# Learn phase -> training of the weight array
def learn(weightTab, error, imageTab) :
    for i in range(LAYER_SIZES[1]):
        for j in range(LAYER_SIZES[0]):
            weightTab[i][j] += (EPSILON * error * imageTab[j])

    return weightTab

def verifNumber(fileName, weightTab) :
    loadFile = readFile(fileName)
    image = loadFile.get('imageTab')
    rightValueSaved = loadFile.get('value')

    potOutput = potentialOutputNeuronCalcul(weightTab, image)
    imageFound = -1
    if (potOutput > 0) :
        imageFound = 1
    else :
        imageFound = 0
    error = int(rightValueSaved) - imageFound
    return error

def verif(weightTab):
    resultFinalError = 0
    errorList = []
    for imageNumber in FILE_VERIF_LIST:
        checkNumberIsRecognized = verifNumber(imageNumber, weightTab)
        resultFinalError += abs(checkNumberIsRecognized)
        errorList.append(checkNumberIsRecognized)

    print(errorList)
    return resultFinalError

def testTrainedModel(weightTabTrained):
    for imageNumber in FILE_TEST_LIST:
        isNumberRecognized = verifNumber(imageNumber, weightTabTrained)
        print(imageNumber,' => ', 'True' if isNumberRecognized == 0 else 'False')

# Function which will train the model and validate or not after each iteration (stop method)
def trainNeuronNetwork(weightTab, cpt, errorTab) :
    choiceFile = randFileChoice()
    loadFile = readFile(choiceFile)
    tab = loadFile.get('imageTab')
    rightValueSaved = loadFile.get('value')

    # Init weight tab only at the beginning when the model is not yet trained
    if (len(weightTab) == 0):
        weightTab = initWeightTab()

    potOutput = potentialOutputNeuronCalcul(weightTab, tab)

    imageFound = -1
    if (potOutput > 0) :
        imageFound = 1
    else :
        imageFound = 0

    # 5 Error calcul
    error = int(rightValueSaved) - imageFound

    # 6 Learn
    newWeight = learn(weightTab, error, tab)

    # Train while models is still making errors
    verifError = verif(newWeight)
    errorTab.append(verifError)
    if (verifError > 0):
        trainNeuronNetwork(newWeight, cpt+1, errorTab)
    else :
        print(cpt)
        print(errorTab)
        plt.plot(errorTab)
        plt.show()

    return newWeight

if __name__ == "__main__":
    trainedWeight = trainNeuronNetwork([], 0, [])
    # At the end we check with verif file (others files not in the training part)
    print('\nVerification PART')
    testTrainedModel(trainedWeight)
