#!/usr/bin/env python3
""" 3. Pooling Back Prop"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='avg'):
    """function that performs back propagation
    over a pooling layer of a neural network"""
    m, h, w, cn = A_prev.shape
    m, hn, wn, cn = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for x in range(hn):
            for y in range(wn):
                for c in range(cn):
                    A_ = A_prev[i, x*sh:kh+x*sh, y*sw:kw+y*sw, c]
                    dA_ = dA_prev[i, x*sh:kh+x*sh, y*sw:kw+y*sw, c]

                    if mode == 'max':
                        """ mask :Array of the same shape as A_,contains a True
                        at the position corresponding to the max entry of A_"""
                        mask = (A_ == np.max(A_))
                        dA_ += np.multiply(dA[i, x, y, c], mask)

                    if mode == 'avg':
                        """Compute the value to distribute on the matrix"""
                        dA_ += dA[i, x, y, c]/(kh*kw)
                        # dA_ += (dA[i, x, y, c])/kh/kw

    return dA_prev
