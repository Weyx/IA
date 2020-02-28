import sys
import argparse
import random
import os
from dotenv import load_dotenv


load_dotenv()

LAYER_SIZES = [48, 1]
EPSILON = 0.01
THETA = 0.5
FILE_TRAIN_LIST = ['zero.txt', 'one.txt']

PATH = os.getenv("PATH_TO_FILE")
# PATH = "/data/LINUX/IA/ProjetPi/"

def readFile(choice):
    imageTab = []
    value = -1
    f= open(PATH+choice,"r")
    # Read file char by char
    while True:
        c = f.read(1)
        if not c:
            # print "End of file"
            break
        if (c == '*'):
            imageTab.append(1)
        if (c == '.'):
            imageTab.append(0)
        if (c == '1' or c == '0'):
            value = c

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

# Randomize the choice of file to be used for training
def randFileChoice():
    return random.choice(FILE_TRAIN_LIST)

# Get the potential of the output neuron
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

def verif(weightTab):
    # Zero check error
    loadFileZero = readFile(FILE_TRAIN_LIST[0])
    imageZero = loadFileZero.get('imageTab')
    potOutputZero = potentialOutputNeuronCalcul(weightTab, imageZero)
    imageFoundZero = -1
    if (potOutputZero > 0) :
        imageFoundZero = 1
    else :
        imageFoundZero = 0
    errorZero = 0 - imageFoundZero

    # One check error
    loadFileOne = readFile(FILE_TRAIN_LIST[1])
    imageOne = loadFileOne.get('imageTab')
    potOutputOne = potentialOutputNeuronCalcul(weightTab, imageOne)
    imageFoundOne = -1
    if (potOutputOne > 0) :
        imageFoundOne = 1
    else :
        imageFoundOne = 0
    errorOne = 1 - imageFoundOne

    print(errorZero, errorOne)
    errorFinal = abs(errorZero) + abs(errorOne)

    return errorFinal



def toBeCalled(weightTab, cpt, errorTab) :
    choiceFile = randFileChoice()
    loadFile = readFile(choiceFile)
    tab = loadFile.get('imageTab')
    rightValueSaved = loadFile.get('value')

    # Init weight tab only at the beginning when the model is not yet trained
    if (len(weightTab) == 0):
        weightTab = initWeightTab()

    potOutput = potentialOutputNeuronCalcul(weightTab, tab)

    imageFound = -1
    if (potOutput >0) :
        imageFound = 1
    else :
        imageFound = 0

    # 5 Error calcul
    error = int(rightValueSaved) - imageFound

    #6 Learn
    newWeight = learn(weightTab, error, tab)

    verifError = verif(newWeight)
    errorTab.append(verifError)
    if (verifError > 0):
        toBeCalled(newWeight, cpt+1, errorTab)
    else :
        print(cpt)
        print(errorTab)



if __name__ == "__main__":
    toBeCalled([], 0, [])
