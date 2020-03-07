import sys
import part3
import matplotlib.pyplot as plt

TRIALS_NUMBER = 100

if __name__ == "__main__":
    # PART 1
    trainedWeight = part3.trainNeuronNetwork([], 0, [])
    # At the end we check with verif file (others files not in the training part)
    print('\nVerification PART')
    errorTab = [[], [], []]
    for i in range (0, 101):
        cpt0 = 0
        cpt1 = 0
        cpt2 = 0
        for j in range(TRIALS_NUMBER):
            nbImageError = part3.testTrainedModel(trainedWeight, i/100)
            cpt0 += nbImageError[0]
            cpt1 += nbImageError[1]
            cpt2 += nbImageError[2]
        errorTab[0].append(cpt0/TRIALS_NUMBER)
        errorTab[1].append(cpt1/TRIALS_NUMBER)
        errorTab[2].append(cpt2/TRIALS_NUMBER)

    # print(errorTab)
    plt.plot(errorTab[0], label="0")
    plt.plot(errorTab[1], label="1")
    plt.plot(errorTab[2], label="2")
    plt.legend(loc='best')
    plt.ylabel('Error (%)')
    plt.xlabel('Noise percentage (%)')
    plt.title('TD1 - 2 : Widrow-Hoff')
    # plt.savefig('generatedPlots/Widrow-Hoff1.png')
    plt.show()
