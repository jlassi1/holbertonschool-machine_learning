#!/usr/bin/env python3
"""5. Gradient Descent with Dropout"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    function  that updates the weights of a neural network with
    Dropout regularization using gradient descent
    """
    m = len(Y[0])
    dZ = cache["A" + str(L)] - Y
    for j in range(L, 0, -1):
        Act = cache["A" + str(j-1)]
        dW = np.matmul(dZ, Act.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        tanh = 1 - Act*Act
        dZ = np.matmul(weights[
            "W" + str(j)].T, dZ) * tanh

        if j > 1:
            dZ *= cache["D"+str(j - 1)]
            dZ /= keep_prob

        weights["W" + str(j)] -= alpha * dW
        weights["b" + str(j)] -= alpha * db
