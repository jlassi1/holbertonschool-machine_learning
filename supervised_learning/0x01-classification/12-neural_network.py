#!/usr/bin/env python3
"""Create a Neuron """
import numpy as np


class NeuralNetwork:
    """defines a neural network with one
    hidden layer performing binary classification"""

    def __init__(self, nx, nodes):
        """Class Initialization"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.nodes = nodes
        self.nx = nx
        self.__W1 = np.random.randn(self.nodes, self.nx)
        self.__A1 = 0
        self.__b1 = np.zeros((self.nodes, 1))
        self.__W2 = np.random.randn(1, self.nodes)
        self.__A2 = 0
        self.__b2 = 0

    @property
    def W1(self):
        """ the getter of the weights"""
        return self.__W1

    @property
    def A1(self):
        """the getter of the activated output"""
        return self.__A1

    @property
    def b1(self):
        """the getter of the bias"""
        return self.__b1

    @property
    def W2(self):
        """ the getter of the weights"""
        return self.__W2

    @property
    def A2(self):
        """the getter of the activated output"""
        return self.__A2

    @property
    def b2(self):
        """the getter of the bias"""
        return self.__b2

    def sigmoid(self, z):
        """ activation function """
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        """ derivative of activation function"""
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward_prop(self, X):
        """the forward propagation of the neuron"""
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(z1)
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(z2)
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = A.shape[1]
        cost = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        C = -(1 / m) * np.sum(cost)
        return C

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        A1, A2 = self.forward_prop(X)
        Y_prediction = np.where(A2 < 0.5, 0, 1)
        return Y_prediction, self.cost(Y, A2)
