import sys
import argparse
import random

LAYER_SIZES = [48, 1]
EPSILON = 0.01
THETA = 0.5
FILE_TRAIN_LIST = ['zero.txt', 'one.txt']
PATH = "/data/LINUX/IA/ProjetPi/"

def readFile(choice):
    tab = []
    value = -1
    f= open(PATH+choice,"r")
    # Read file char by char
    while True:
        c = f.read(1)
        if not c:
            # print "End of file"
            break
        if (c == '*'):
            tab.append(1)
        if (c == '.'):
            tab.append(0)
        if (c == '1' or c == '0'):
            value = c
        
    dictionnary = dict()
    dictionnary['tab'] = tab
    dictionnary['value'] = value
    return dictionnary
            
def initWeightTab(): 
    tab = []
    # Init tab
    for i in range(LAYER_SIZES[1]):
        tab.append([0] * LAYER_SIZES[0])

    for i in range(LAYER_SIZES[1]):
        for j in range(LAYER_SIZES[0]):
            rnd = random.random()
            tab[i][j] = rnd  / 48

    # print(tab)
    return tab

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


# def toBeCalled(weightTab, cpt, wrong) : 
#     choiceFile = randFileChoice()
#     # print(choiceFile)
#     loadFile = readFile(choiceFile)
#     tab = loadFile.get('tab')
#     rightValueSaved = loadFile.get('value')
#     # print(tab)

#     if (len(weightTab) == 0):
#         weightTab = initWeightTab()
#     # print(weightTab)
#     # print('\n')
#     # print(len(weightTab))

#     potOutput = potentialOutputNeuronCalcul(weightTab, tab)
#     # print(potOutput)

#     imageFound = -1
#     if (potOutput >0) : 
#         imageFound = 1
#     else : 
#         imageFound = 0

#     if (imageFound != int(rightValueSaved)):
#         wrong += 1
#     print(str(imageFound)," ",rightValueSaved)

#     # 5 Error calcul
#     error = int(rightValueSaved) - imageFound

#     #6 Learn
#     # print(weightTab)
#     newWeight = learn(weightTab, error, tab)
    
#     # print('\n')
#     # print(newWeight)

#     if (cpt < 100):
#         toBeCalled(newWeight, cpt+1, wrong)
#     else:
#         print(wrong)

def verif(weightTab):
    # Zero check error
    loadFileZero = readFile(FILE_TRAIN_LIST[0])
    imageZero = loadFileZero.get('tab')
    potOutputZero = potentialOutputNeuronCalcul(weightTab, imageZero)
    imageFoundZero = -1
    if (potOutputZero >0) : 
        imageFoundZero = 1
    else : 
        imageFoundZero = 0
    errorZero = 0 - imageFoundZero

    # One check error
    loadFileOne = readFile(FILE_TRAIN_LIST[1])
    imageOne = loadFileOne.get('tab')
    potOutputOne = potentialOutputNeuronCalcul(weightTab, imageOne)
    imageFoundOne = -1
    if (potOutputOne >0) : 
        imageFoundOne = 1
    else : 
        imageFoundOne = 0
    errorOne = 1 - imageFoundOne

    errorFinal = abs(errorZero) + abs(errorOne)

    return errorFinal



def toBeCalled(weightTab, cpt, errorTab) :
    choiceFile = randFileChoice()
    loadFile = readFile(choiceFile)
    tab = loadFile.get('tab')
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
    # print(str(imageFound)," ",rightValueSaved)

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

    
