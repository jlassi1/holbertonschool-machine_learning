#!/usr/bin/env python3
"""1. Pooling Forward Prop"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """function that performs forward propagation
    over a pooling layer of a neural network"""
    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    """ output size"""
    nh = int((h-kh)/sh + 1)
    nw = int((w-kh)/sw + 1)
    output = np.empty((m, nh, nw, c))

    for x in range(nh):
        for y in range(nw):
            if mode == 'max':
                output[:, x, y, :] = np.max(
                    A_prev[:, x*sh:kh+x*sh, y*sw:kw+y*sw, :],
                    axis=(1, 2))
            if mode == 'avg':
                output[:, x, y, :] = np.average(
                    A_prev[:, x*sh:kh+x*sh, y*sw:kw+y*sw, :],
                    axis=(1, 2))
    return output
