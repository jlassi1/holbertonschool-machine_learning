#!/usr/bin/env python3
"""Create a Neuron """
import numpy as np


class Neuron:
    """defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Class Initialization """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = np.random.randn(1, self.nx)
        self.__A = 0
        self.__b = 0

    @property
    def W(self):
        """ the getter of the weights"""
        return self.__W

    @property
    def A(self):
        """the getter of the activated output"""
        return self.__A

    @property
    def b(self):
        """the getter of the bias"""
        return self.__b

    def sigmoid(self, z):
        """ activation function """
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        """ derivative of activation function"""
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward_prop(self, X):
        """the forward propagation of the neuron"""
        z = np.dot(self.__W, X) + self.__b
        self.__A = self.sigmoid(z)
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = A.shape[1]
        cost = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        C = -(1 / m) * np.sum(cost)
        return C
