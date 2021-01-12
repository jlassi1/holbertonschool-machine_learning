#!/usr/bin/env python3
import numpy as np


def shuffle_data(X, Y):
    """function that shuffles the data points in two matrices the same way"""
    randomize = np.random.permutation(X.shape[0])
    return X[randomize], Y[randomize]
