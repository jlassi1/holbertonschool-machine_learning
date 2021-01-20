#!/usr/bin/env python3
"""4. Forward Propagation with Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    function that conducts forward propagation using Dropout
    """
    cache = {}
    cache["A0"] = X
    for j in range(L):
        z = np.dot(weights["W" + str(j+1)], cache[
            "A" + str(j)]) + (weights["b" + str(j+1)])
        cache["A" + str(j+1)] = np.tanh(z)/keep_prob
    return cache
