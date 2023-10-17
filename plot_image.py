import matplotlib.pyplot as pyplot
import csv
import numpy

data=[]

with open('/home/tony/programming/deep_autoencoder/model_images.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)

images = []
for i in range(0,len(data),2):
    images.append((data[i], data[i+1]))


for i in images:
    f, (ax1, ax2) = pyplot.subplots(1,2)
    pyplot.suptitle(f'Epoch: {i[0][0]}')
    before = numpy.array(i[0][1:], dtype=float)
    before = before.reshape((28,28))
    ax1.imshow(before, cmap='gray')
    ax1.set_title("Input:")
    after = numpy.array(i[1][1:], dtype=float)
    after = after.reshape((28,28))
    ax2.imshow(after, cmap='gray')
    ax2.set_title("Output:")
    pyplot.show()