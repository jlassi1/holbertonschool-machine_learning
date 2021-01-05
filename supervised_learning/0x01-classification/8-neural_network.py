#!/usr/bin/env python3
"""Create a NeuralNetwork """
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
        self.W1 = np.random.randn(self.nodes, self.nx)
        self.A1 = 0
        self.b1 = np.zeros((self.nodes, 1))
        self.W2 = np.random.randn(1, self.nodes)
        self.A2 = 0
        self.b2 = 0
