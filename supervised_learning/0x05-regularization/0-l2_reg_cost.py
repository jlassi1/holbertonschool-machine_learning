#!/usr/bin/env python3
""" 0. L2 Regularization Cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    function that calculates the cost
    of a neural network with L2 regularization
    """
    _norm = np.linalg.norm
    ws = []
    for i in range(L):
        ws.append(_norm(weights["W"+str(i+1)]))

    L2 = cost + (lambtha/(2*m))*sum(ws)
    return L2
