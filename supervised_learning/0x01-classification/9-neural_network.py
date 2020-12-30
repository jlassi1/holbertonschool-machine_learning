#!/usr/bin/env python3
"""Create a Neuron """
import numpy as np


class NeuralNetwork:
    """defines a neural network with one hidden layer performing binary classification"""

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
        self.__W2 = np.random.randn(1,len(self.W1))
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
