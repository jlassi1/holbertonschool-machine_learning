#!/usr/bin/env python3
""" 0. RNN Cell"""
import numpy as np


class RNNCell:
    """ class that represents a cell of a simple RNN"""
    def __init__(self, i, h, o):
        """ initialization"""

        # Weights
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        # Biases
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """function that performs forward propagation for one time step"""
        xh = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(xh, self.Wh) + self.bh)
        y = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y)
        return h_next, y