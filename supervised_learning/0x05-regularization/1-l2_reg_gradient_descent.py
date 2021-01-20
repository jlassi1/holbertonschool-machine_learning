#!/usr/bin/env python3
"""1. Gradient Descent with L2 Regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    updates the weights and biases of a neural network
    using gradient descent with L2 regularization
    """
    _norm = np.linalg.norm
    m = len(Y[0])
    dZ = cache["A" + str(L)] - Y
    for j in range(L, 0, -1):
        Act = cache["A" + str(j-1)]
        dW = np.matmul(dZ, Act.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dZ = np.matmul(weights[
            "W" + str(j)].T, dZ) * np.tanh(Act)

        dW_L2reg = dW + (lambtha/m)*weights["W" + str(j)]
        #db_L2reg = db + (lambtha/m)*weights["b" + str(j)]

        weights["W" + str(j)] -= alpha * dW_L2reg
        weights["b" + str(j)] -= alpha * db
