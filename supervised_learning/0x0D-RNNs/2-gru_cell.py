#!/usr/bin/env python3
"""2. GRU Cell"""
import numpy as np


class GRUCell:
    """ class that represents a gated recurrent unit"""

    def __init__(self, i, h, o):
        """ initialization"""
        # Weights
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        # Biases
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def sigmoid(self, x):
        """Compute sigmoid values for each sets of scores in x."""
        return np.exp(-np.logaddexp(0, -x))

    def forward(self, h_prev, x_t):
        """function that performs forward propagation for one time step"""
        xh = np.concatenate((h_prev, x_t), axis=1)
        # 1. Update gate
        z_t = self.sigmoid(np.matmul(xh, self.Wz) + self.bz)
        # 2. Reset gate
        r_t = self.sigmoid(np.matmul(xh, self.Wr) + self.br)

        # 3. Current memory content
        xhr = np.hstack(((r_t * h_prev), x_t))
        H_tilda = np.tanh(np.dot(xhr, self.Wh) + self.bh)

        # 4. Final memory at current time step
        h_next = z_t * H_tilda + (1 - z_t) * h_prev

        # 5. compte the output of the cell
        y = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y)

        return h_next, y
