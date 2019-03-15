# MNIST Data
# https://pjreddie.com/projects/mnist-in-csv/
# Training set: 60,000 examples
# Test set: 10,000 examples
# label, 28x28 values (784)

import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork as NN

# training data input
train_src = open("D:\Articles\MN_NN\data\mnist_train.csv", 'r')
training_data = train_src.readlines()
train_src.close()

# test data input
test_src = open("D:\Articles\MN_NN\data\mnist_test.csv", 'r')
test_data = test_src.readlines()
test_src.close()

# plotting single number from training
def show_example(data, x):
    img_array = np.asfarray(data[x].split(',')[1:]).reshape((28,28))
    plt.imshow(img_array, cmap='Greys', interpolation='None')
    plt.show()

# Number of input, hidden, and output nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# Learning rate
learning_rate = 0.1

# Initialise Neural network
N = NN(input_nodes, hidden_nodes, output_nodes, learning_rate)

# TRAINING
epochs = 2

for epoch in range(epochs):
    for sample in training_data:
        sample_values = sample.split(',')
        # scale and shift input
        inputs = (np.asfarray(sample_values[1:]) / 255.0 * 0.99) + 0.01
        # create target values (0.01 -> 0.99)
        targets = np.zeros(output_nodes) + 0.01
        # assign desired target values
        targets[int(sample_values[0])] = 0.99
        N.train(inputs, targets)

# TEST Neural network

scores = []
for sample in test_data:
    sample_values = sample.split(',')
    # Correct labes is first
    correct_label = int(sample_values[0])
    # scale and shift inputs
    inputs = (np.asfarray(sample_values[1:]) / 255.0 * 0.99) + 0.01
    # Query NN
    outputs = N.query(inputs)
    # index of highest output is our label
    label = np.argmax(outputs)

    # append label to scores
    if label == correct_label:
        # predicted label is correct
        scores.append(1)
    else:
        scores.append(0)

# Measure performance of NN
performance = np.asarray(scores)
print('Performance = ', performance.sum() / float(performance.size))
# 95.03
