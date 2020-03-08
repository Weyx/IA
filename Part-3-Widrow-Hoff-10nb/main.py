import sys
import part3
import matplotlib.pyplot as plt

TRIALS_NUMBER = 100
NUMBER_NB = 10

if __name__ == "__main__":
    # PART 1
    trainedWeight = part3.trainNeuronNetwork([], 0, [])
    # At the end we check with verif file (others files not in the training part)
    print('\nVerification PART')
    errorTab = [[]] * NUMBER_NB

    for i in range (0, 101):
        cpt = [0] * NUMBER_NB
        for j in range(TRIALS_NUMBER):
            nbImageError = part3.testTrainedModel(trainedWeight, i/100)
            print(nbImageError)
            for k in range(NUMBER_NB):
                cpt[k] += nbImageError[k]

        for w in range(NUMBER_NB):
            errorTab[w].append(cpt[w]/TRIALS_NUMBER)

    # print(errorTab)
    # for y in range(NUMBER_NB):
    #     plt.plot(errorTab[y], label=str(y))
    plt.plot(errorTab[0], label="0")
    # plt.plot(errorTab[1], label="1")
    # plt.plot(errorTab[2], label="2")
    plt.legend(loc='best')
    plt.ylabel('Error (%)')
    plt.xlabel('Noise percentage (%)')
    plt.title('TD1 - 2 : Widrow-Hoff')
    # plt.savefig('generatedPlots/Widrow-Hoff1.png')
    plt.show()
