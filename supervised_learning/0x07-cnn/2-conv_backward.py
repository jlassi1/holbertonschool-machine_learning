#!/usr/bin/env python3
"""2. Convolutional Back Prop """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """function that performs back propagation over
    a convolutional layer of a neural network"""
    m, h, w, cp = A_prev.shape
    m, hn, wn, cn = dZ.shape
    kh, kw, cp, cn = W.shape
    sh, sw = stride
    """ padding condition"""
    # if padding == 'valid':
    ph, pw = 0, 0
    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    """ output size"""
    nh = int((h-kh+2*ph)/sh + 1)
    nw = int((w-kw+2*pw)/sw + 1)
    output = np.empty((m, nh, nw, cn))
    """ create a pad layer"""
    A = np.pad(A_prev, pad_width=((0,), (ph,), (pw,), (0,)),
               mode="constant",
               constant_values=0)
    """ Initialize dA, dW, db with the correct shapes"""
    dW = np.zeros_like(W)
    dA = np.zeros_like(A)
    db = np.zeros_like(b)
    """ update  the biases db=∑h∑wdZhw"""
    # db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):
        for x in range(nh):
            for y in range(nw):
                for c in range(cn):
                    A_ = A[i, x*sh:kh+x*sh, y*sw:kw+y*sw, :]
                    dA_ = dA[i, x*sh:kh+x*sh, y*sw:kw+y*sw, :]
                    dA_ += dZ[i, x, y, c]*W[:, :, :, c]
                    dW[:, :, :, c] += A_ * dZ[i, x, y, c]
                    db[:, :, :, c] += dZ[i, x, y, c]

    # dA without padding
    # if padding == 'same':
    dA = dA[:, ph:-ph, pw:-pw, :]
    # else:
    #    dA = dA
    return dA, dW, db
