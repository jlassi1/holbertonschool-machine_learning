#!/usr/bin/env python3
"""Create a DeepNeuralNetwork """
import numpy as np


class DeepNeuralNetwork:
    """defines a deep neural network
    performing binary classification"""

    def __init__(self, nx, layers):
        """Class Initialization"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or layers == []:
            raise TypeError("layers must be a list of positive integers")
        elif all([x <= 0 for x in layers]):
            raise ValueError("layers must be a list of positive integers")
        self.layers = layers
        self.L = len(layers) - 1
        self.cache = {}
        weights = {}
        for l in range(len(layers)):
            if l == 0:
                weights["W" + str(l + 1)] = np.random.randn(
                    layers[l], nx) * np.sqrt(2 / nx)

                weights["b" + str(l + 1)] = np.zeros((layers[l], 1))
            else:
                weights["W" + str(l + 1)] = np.random.randn(
                    layers[l], layers[l-1]) * np.sqrt(2/layers[l-1])

                weights["b" + str(l + 1)] = np.zeros((layers[l], 1))
        self.weights = weights
