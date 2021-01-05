#!/usr/bin/env python3
"""Create a DeepNeuralNetwork """
import numpy as np


class DeepNeuralNetwork:
    """defines a deep neural network
    performing binary classification"""

    def __init__(self, nx, layers):
        """ Class Initialization """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or layers == []:
            raise TypeError("layers must be a list of positive integers")
        if not isinstance(all(layers), int) or not (all(layers) > 0):
            raise TypeError("layers must be a list of positive integers")
        self.__layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for j in range(len(layers)):
            if j == 0:
                self.__weights["W" + str(j + 1)] = np.random.randn(
                    layers[j], nx) * np.sqrt(2 / nx)

                self.__weights["b" + str(j + 1)] = np.zeros((layers[j], 1))
            else:
                self.__weights["W" + str(j + 1)] = np.random.randn(
                    layers[j], layers[j - 1]) * np.sqrt(2 / layers[j - 1])

                self.__weights["b" + str(j + 1)] = np.zeros((layers[j], 1))

    @property
    def layers(self):
        return self.__layers

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def sigmoid(self, z):
        """ activation function """
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        """ derivative of activation function"""
        return z * (1 - z)

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache["A0"] = X
        for j in range(self.__L):
            s = np.matmul(self.__weights["W" + str(j+1)], self.__cache[
                "A" + str(j)]) + (self.__weights["b" + str(j+1)])
            self.__cache["A" + str(j+1)] = self.sigmoid(s)
        return self.__cache["A" + str(self.__L)], self.__cache
