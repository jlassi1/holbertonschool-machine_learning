#!/usr/bin/env python3
""" 1. RNN """
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """function  that performs forward propagation for a bidirectional RNN"""
    # Get shapes
    T, m, i = X.shape
    o = bi_cell.by.shape[1]
    h = h_0.shape[1]
    Hf = np.empty((T + 1, m, h))
    Hb = np.empty((T + 1, m, h))
    Y = np.empty((T, m, o))
    Hf[0] = h_0
    Hb[-1] = h_t
    for t in range(1, T + 1):
        Hf[t] = bi_cell.forward(Hf[t - 1], X[t - 1])
    for t in range(0, T)[::-1]:
        Hb[t] = bi_cell.backward(Hb[t + 1], X[t])

    H = np.concatenate((Hf[1:], Hb[:-1]), axis=-1)
    Y = bi_cell.output(H)
    return H, Y
