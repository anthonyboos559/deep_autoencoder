import matplotlib.pyplot as pyplot
import csv

data = []
epochs = [i for i in range(1,31)]

with open('/home/tony/programming/deep_autoencoder/model_data.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)

train = [float(i) for i in data[0]]
test = [float(i) for i in data[1]]


pyplot.plot(epochs, train, 's-r', linewidth=2, label='Train set error')
pyplot.plot(epochs, test, 's-b', linewidth=2, label='Test set error')
pyplot.axis([0,30,20,60])
pyplot.title("Model Performance")
pyplot.xlabel("Epochs")
pyplot.ylabel("Mean Squared Error")
pyplot.legend(loc="best")

pyplot.show()