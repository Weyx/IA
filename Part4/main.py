import sys
import matplotlib.pyplot as plt
import gzip
import numpy as np

imageFile = gzip.open('./samples/train-images-idx3-ubyte.gz','r')
imageFile.read(16)
labelFile = gzip.open('./samples/train-labels-idx1-ubyte.gz','r')
labelFile.read(8)

def ascii_show(image):
    print("\n\n")
    for y in image:
        row = ""
        for x in y:
            row += '{0: <4}'.format(x)
        print(row)

def readNewImage():
    # READ IMAGE (28x28)
    imageSize = 28
    nbImages = 1

    buf = imageFile.read(imageSize * imageSize * nbImages)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(nbImages, imageSize, imageSize, 1)

    image = np.asarray(data[0]).squeeze()
    ascii_show(image)

    # READ LABELS (1, 2, 3, ..., 9)
    buf = labelFile.read(1)
    label = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    print(label)

    returnedValues = dict()
    returnedValues['imageTab'] = image
    returnedValues['label'] = label[0]
    return returnedValues

if __name__ == "__main__":
    for i in range(1):
        returnedValue = readNewImage()
        # print(returnedValue.get("label"))
        # print(returnedValue.get("imageTab"))

#https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python => first url used to read in gz file and extract data
