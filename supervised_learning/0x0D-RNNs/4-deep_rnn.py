#!/usr/bin/env python3
""" 4. Deep RNN """
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """function that performs forward propagation for a deep RNN"""
    # Get shapes
    length = len(rnn_cells)
    l, m, h = h_0.shape
    t, m, i = X.shape
    o = rnn_cells[-1].by.shape[1]
    H = np.empty((t + 1, length, m, h))
    Y = np.empty((t, m, o))
    H[0] = h_0

    for t in range(1, t + 1):
        for i in range(1, length):
            H[t, 0], Y[t - 1] = rnn_cells[0].forward(H[t - 1, 0], X[t - 1])
            cell = rnn_cells[i]
            H[t, i], Y[t - 1] = cell.forward(H[t - 1, i], H[t, i - 1])
    return H, Y
