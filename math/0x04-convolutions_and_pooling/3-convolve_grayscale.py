#!/usr/bin/env python3
""" convolve function """
import numpy as np


def convolve_grayscale(images, kernel, padding='same',
                       stride=(1, 1)):
    """ Function that that performs a convolution on
        grayscale images """
    m, h, w = np.shape(images)
    kh, kw = np.shape(kernel)
    sh, sw = stride
    if padding == "valid":
        ph = 0
        pw = 0
    elif padding == "same":
        """ph = (kh-kh % 2)//2
        pw = (kw-kw % 2)//2"""
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    else:
        ph, pw = padding
    image = np.zeros((m, (h + 2 * ph), (w + 2 * pw)))
    image[:, ph:h+ph, pw:w+pw] = images.copy()
    nh = np.floor(((h + 2 * ph - kh) / stride[0]) + 1).astype(int)
    nw = np.floor(((w + 2 * pw - kw) / stride[1]) + 1).astype(int)
    S = np.zeros((m, nh, nw))
    im = np.arange(0, m)
    for i in range(nh):
        for j in range(nw):
            S[im, i, j] = np.sum(image[im,
                                       sh * i:sh * i + kh,
                                       sw * j:sw * j + kw] * kernel,
                                 axis=(1, 2))
    return S
