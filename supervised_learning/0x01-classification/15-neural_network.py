#!/usr/bin/env python3
"""Create a Neuron """
import numpy as np
import matplotlib.pyplot as plt


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
        m = X.shape[1]
        dz2 = A2 - Y
        dz1 = np.matmul(self.__W2.T, dz2) * self.sigmoid_derivative(A1)

        dW1 = np.matmul(dz1, X.T) / m
        self.__W1 = self.__W1 - alpha * dW1

        dW2 = np.matmul(dz2, A1.T) / m
        self.__W2 = self.__W2 - alpha * dW2

        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        self.__b1 = self.__b1 - alpha * db1

        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        self.__b2 = self.__b2 - alpha * db2

    def train(
        self, X, Y,
        iterations=5000, alpha=0.05,
        verbose=True, graph=True, step=100
            ):
        """ Trains the neural network """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("tep must be positive and <= iterations")
        cost = []
        iter = []
        for iteration in range(iterations + 1):
            self.__A1, self.__A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            if verbose and iteration % 100 == 0:
                cost.append(self.cost(Y, self.__A2))
                iter.append(iteration)
                print("Cost after {} iterations: {}".format(
                    iteration, self.cost(Y, self.__A2)))
        if graph:
            plt.plot(iter, cost)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
