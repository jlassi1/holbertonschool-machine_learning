#!/usr/bin/env python3
"""Create a Neuron """
import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions"""
        A = self.forward_prop(X)
        Y_prediction = np.where(A < 0.5, 0, 1)
        return Y_prediction, self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = X.shape[1]

        dW = np.dot((A - Y), X.T) / m
        self.__W = self.__W - alpha * dW

        db = np.sum((A - Y)) / m
        self.__b = self.__b - alpha * db
        return self.__W, self.__b

    def train(
            self,
            X,
            Y,
            iterations=5000,
            alpha=0.05,
            verbose=True,
            graph=True,
            step=100):
        """Trains the neuron"""
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
                raise ValueError("step must be positive and <= iterations")
        cost = []
        iter = []
        for iteration in range(iterations + 1):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)

            if verbose and iteration % 100 == 0:
                cost.append(self.cost(Y, self.__A))
                iter.append(iteration)
                print("Cost after {} iterations: {}".format(
                    iteration, self.cost(Y, self.__A)))
        if graph:
            plt.plot(iter, cost)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
