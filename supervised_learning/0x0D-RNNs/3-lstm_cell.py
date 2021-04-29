#!/usr/bin/env python3
"""3. LSTM Cell """
import numpy as np


class LSTMCell:
    """ class that represents an LSTM unit"""

    def __init__(self, i, h, o):
        """ initialization"""
        # Weights
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        # Biases
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def sigmoid(self, x):
        """Compute sigmoid values for each sets of scores in x."""
        return np.exp(-np.logaddexp(0, -x))

    def forward(self, h_prev, c_prev, x_t):
        """function that performs forward propagation for one time step"""
        xh = np.concatenate((h_prev, x_t), axis=1)
        # 1. Forget gate
        f_t = self.sigmoid(np.matmul(xh, self.Wf) + self.bf)
        # 2.  Update gate
        u_t = self.sigmoid(np.matmul(xh, self.Wu) + self.bu)

        # 3.  Updating the cell
        c_t = np.tanh(np.matmul(xh, self.Wc) + self.bc)
        c_next = f_t*c_prev + u_t*c_t

        # 4. Output gate
        o_t = self.sigmoid(np.matmul(xh, self.Wo) + self.bo)
        h_next = o_t * np.tanh(c_next)

        # 5. compte the output of the cell
        y = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y)

        return h_next, c_next, y
