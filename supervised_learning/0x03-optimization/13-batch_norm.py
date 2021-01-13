#!/usr/bin/env python3
""" 13. Batch Normalization """
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """function that normalizes an unactivated output of
    a neural network using batch normalization"""
    m = np.mean(Z, axis=0)
    s = np.std(Z, axis=0)
    Z_norm = (Z - m) / s
    Z_ = gamma*Z_norm+beta
    return Z_
