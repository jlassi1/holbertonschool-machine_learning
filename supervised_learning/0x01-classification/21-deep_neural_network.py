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
            s = np.dot(self.__weights["W" + str(j+1)], self.__cache[
                "A" + str(j)]) + (self.__weights["b" + str(j+1)])
            self.__cache["A" + str(j+1)] = self.sigmoid(s)
        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = A.shape[1]
        cost = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        C = -(1 / m) * np.sum(cost)
        return C

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        self.forward_prop(X)
        Y_prediction = np.where(self.__cache["A" + str(self.__L)] < 0.5, 0, 1)
        return Y_prediction, self.cost(Y, self.__cache["A" + str(self.__L)])

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = len(Y[0])
        dZ = cache["A" + str(self.__L)] - Y
        for j in range(self.__L, 0, -1):
            A = cache["A" + str(j-1)]
            dW = np.matmul(dZ, A.T) / m
            self.__weights["W"+str(j)] -= alpha * dW

            db = np.sum(dZ, axis=1, keepdims=True) / m
            self.__weights["b"+str(j)] -= alpha * db

            dZ = np.matmul(self.__weights[
                "W"+str(j)].T, dZ) * self.sigmoid_derivative(A)
