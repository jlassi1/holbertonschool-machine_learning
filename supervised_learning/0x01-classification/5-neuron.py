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
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(- z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = A.shape[1]
        cost = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        total_cost = -(1 / m) * np.sum(cost)
        return total_cost

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions"""
        m = X.shape[1]
        Y_prediction = np.zeros((1,m), dtype=int)
        
        for i in range(Y.shape[1]):
            if X[0, i] >= 0.5:
                Y_prediction[0, i] = 1
            else:
                Y_prediction[0, i] = 0

        return Y_prediction, self.cost(Y, self.forward_prop(X))

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        