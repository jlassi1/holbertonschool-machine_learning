#!/usr/bin/env python3
""" 1. RNN """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """function that performs forward propagation for a simple RNN"""
    # Get shapes
    t, m, i = X.shape
    m, h = h_0.shape
    o = rnn_cell.Wy.shape[1]
    # print(o)
    # declare the output (results)
    H = np.empty(shape=(t + 1, m, h))
    Y = np.empty(shape=(t, m, o))
    # Initialize the first hidden layer
    H[0] = h_0
    # loop over all time-steps
    for s in range(t):
        # Update next hidden state, compute the prediction y
        h_0, y = rnn_cell.forward(h_0, X[s, :, :])
        H[s + 1, :, :] = h_0
        Y[s, :, :] = y
    return H, Y
