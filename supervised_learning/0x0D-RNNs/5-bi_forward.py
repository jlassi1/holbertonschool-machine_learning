#!/usr/bin/env python3
""" 5. Bidirectional Cell Forward """
import numpy as np


class BidirectionalCell:
    """ class that represents a bidirectional cell of an RNN"""

    def __init__(self, i, h, o):
        """ initialization"""
        # Weights
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2*h, o)

        # Biases
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def sigmoid(self, x):
        """Compute sigmoid values for each sets of scores in x."""
        return np.exp(-np.logaddexp(0, -x))

    def forward(self, h_prev, x_t):
        """function that calculates the hidden state
        in the forward direction for one time step """
        xh = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.matmul(xh, self.Whf) + self.bhf)
        return h_next
