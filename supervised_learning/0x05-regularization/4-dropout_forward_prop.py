#!/usr/bin/env python3
"""4. Forward Propagation with Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    function that conducts forward propagation using Dropout
    """
    rnd = np.random.rand
    cache = {}
    cache["A0"] = X
    for j in range(L):
        z = np.dot(weights["W" + str(j+1)], cache[
            "A" + str(j)]) + (weights["b" + str(j+1)])
        if j == L-1:
            """ The last layer use the softmax activation function"""
            ACT = np.exp(z)/sum(np.exp(z))
        else:
            """All layers except the last use the tanh activation function"""
            ACT = np.tanh(z)
            d3 = (rnd(ACT.shape[0], ACT.shape[1]) < keep_prob).astype(int)
            """ACT = np.multiply(ACT, d3)"""
            ACT *= d3
            ACT /= keep_prob
            cache["D"+str(j+1)] = d3

        cache["A"+str(j+1)] = ACT
    return cache
