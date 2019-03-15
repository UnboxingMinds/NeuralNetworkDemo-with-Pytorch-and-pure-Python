"""
"""
import numpy as np
from scipy.special import expit as sigmoid


class NeuralNetwork(object):
    '''
    Compact 3 layer neural network
    (input -> hidden -> output)
    '''

    def __init__(self, input_nodes, hidder_layers, output_nodes, learning_rate):
        self.i_nodes = input_nodes
        self.h_nodes = hidder_layers
        self.o_nodes = output_nodes
        self.alpha = learning_rate

        # Weight matrix between input -> hidden -> output
        # Matrix multiplication: input * hidden * output
        self.wih = np.random.rand(self.h_nodes, self.i_nodes) - 0.5
        self.who = np.random.rand(self.o_nodes, self.h_nodes) - 0.5

        # Normal dist initialisation
        # self.wih = np.random.normal(0.0, pow(self.i_nodes, -0.5), (self.h_nodes, self.i_nodes))
        # self.who = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.o_nodes, self.h_nodes))

        # Activation function expit - sigmoid
        self.activation_func = lambda x: sigmoid(x)

    def train(self, input_list, target_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        # signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # signals from hidden layer
        hidden_outputs = self.activation_func(hidden_inputs)
        # signals into output
        final_inputs = np.dot(self.who, hidden_outputs)
        # signals from output
        outputs = self.activation_func(final_inputs)

        # Errors
        # error between hidden and output
        output_errors = targets - outputs
        # error between input and hiddent
        hidden_errors = np.dot(self.who.T, output_errors)

        # Updateing weights
        # updating weight between hidden and output layers
        self.who += self.alpha * np.dot((output_errors * outputs * (1.0 - outputs)),
                                        np.transpose(hidden_outputs))
        # updating weight between input and hidden
        self.wih += self.alpha * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        np.transpose(inputs))

    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T

        # signals into hidden input
        hidden_inputs = np.dot(self.wih, inputs)
        # signals from hidden input
        hidden_outputs = self.activation_func(hidden_inputs)

        # signals into output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # signals from output layer
        outputs = self.activation_func(final_inputs)

        return outputs
