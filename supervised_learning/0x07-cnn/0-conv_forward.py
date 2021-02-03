#!/usr/bin/env python3
""" convolve function """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ Function that that performs forward propagation
        over a convolutional layer of a neural network """
    m, h, w, c = np.shape(A_prev)
    kh, kw, c, nc = np.shape(W)
    sh, sw = stride
    if padding == "valid":
        ph = 0
        pw = 0
    elif padding == "same":
        ph = int(((h - 1) * sh + kh - kh % 2 - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - kw % 2 - w) / 2) + 1
    else:
        ph, pw = padding
    image = np.zeros((m, (h + 2 * ph), (w + 2 * pw), c))
    image[:, ph:h+ph, pw:w+pw, :] = A_prev.copy()
    nh = np.floor(((h + 2 * ph - kh) / stride[0]) + 1).astype(int)
    nw = np.floor(((w + 2 * pw - kw) / stride[1]) + 1).astype(int)
    S = np.zeros((m, nh, nw, nc))
    im = np.arange(0, m)
    for i in range(nh):
        for j in range(nw):
            for k in range(nc):
                S[im, i, j, k] += np.sum(image[im, sh * i:sh * i + kh,
                                               sw * j:sw * j + kw, :]
                                         * W[:, :, :, k],
                                         axis=(1, 2, 3))
    return activation(S + b)
