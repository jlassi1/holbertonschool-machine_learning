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

    def forward_prop(self, X):
        """the forward propagation of the neuron"""
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(- z))
        return self.__A
