#!/usr/bin/env python3
"""0. Convolutional Forward Prop"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """function  that performs forward propagation
    over a convolutional layer of a neural network"""
    m, h, w, c = A_prev.shape
    kh, kw, cp, cn = W.shape
    sh, sw = stride
    """ padding condition"""
    if padding == 'valid':
        ph = 0
        pw = 0
    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))

    """ output size"""
    nh = int((h-kh+2*ph)/sh + 1)
    nw = int((w-kw+2*pw)/sw + 1)
    output = np.zeros((m, nh, nw, cn))
    """ create a pad layer"""
    pad_lay = np.pad(A_prev, pad_width=((0,), (ph,), (pw,), (0,)),
                     mode="constant",
                     constant_values=0)

    for x in range(nh):
        for y in range(nw):
            for c in range(cn):
                slice_ = pad_lay[:, x*sh:kh+x*sh, y*sw:kw+y*sw, :]
                output[:, x, y, c] = activation(np.sum(
                    slice_*W[:, :, :, c], axis=(1, 2, 3)) + b[:, :, :, c])

    return output
