#!/usr/bin/env python3
"""Create a Neuron """
import numpy as np


class NeuralNetwork:
    """defines a neural network with one
    hidden layer performing binary classification"""

    def __init__(self, nx, nodes):
        """Class Initialization """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.nx = nx
        self.__W1 = np.random.randn(3, self.nx)
        self.__A1 = 0
        self.__b1 = np.zeros((len(self.W1), 1))
        self.__W2 = np.random.randn(1, len(self.W1))
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
        return z * (1 - z)

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

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network """
        DZ2 = A2 - Y
        deriv_sigmoid = A1 * (1 - A1)
        m = Y.shape[1]
        DZ1 = np.matmul(self.W2.T, DZ2) * deriv_sigmoid
        self.__W2 = self.__W2 - alpha * np.matmul(DZ2, A1.T) / m
        self.__b2 = self.__b2 - alpha * np.sum(DZ2, axis=1, keepdims=True) / m
        self.__W1 = self.__W1 - alpha * np.matmul(DZ1, X.T) / m
        self.__b1 = self.__b1 - alpha * np.sum(DZ1, axis=1, keepdims=True) / m
