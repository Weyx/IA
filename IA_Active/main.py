import sys
import part1, part1_variante
import matplotlib.pyplot as plt

TRIALS_NUMBER = 50

if __name__ == "__main__":
    # PART 1
    trainedWeight = part1.trainNeuronNetwork([], 0, [])
    # At the end we check with verif file (others files not in the training part)
    print('\nVerification PART')
    errorTab = [[], []]
    for i in range (0, 101):
        cpt0 = 0
        cpt1 = 0
        for j in range(TRIALS_NUMBER):
            nbImageError = part1.testTrainedModel(trainedWeight, i/100)
            cpt0 += nbImageError[0]
            cpt1 += nbImageError[1]
        errorTab[0].append(cpt0)
        errorTab[1].append(cpt1)

    print(errorTab)
    plt.plot(errorTab[0], label="0")
    plt.plot(errorTab[1], label="1")
    plt.legend(loc='best')
    plt.ylabel('Error number in image recognition')
    plt.xlabel('Noise percentage (%)')
    plt.title('TD1 - 1 : Normal')
    # plt.savefig('generatedPlots/test.png')
    plt.show()


    # PART 1 - VARIANTE
    # trainedWeight = part1_variante.trainNeuronNetwork([], 0, [])
    # # At the end we check with verif file (others files not in the training part)
    # print('\nVerification PART')
    # part1_variante.testTrainedModel(trainedWeight)
