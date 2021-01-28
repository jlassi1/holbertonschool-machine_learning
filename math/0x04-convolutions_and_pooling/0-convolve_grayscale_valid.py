#!/usr/bin/env python3
"""  0. Valid Convolution """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """function  that performs a valid convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    final_h = h - kh + 1
    final_w = w - kw + 1
    output = np.zeros((m, final_h, final_w))
    for x in range(final_h):
        for y in range(final_w):
            output[:, x, y] = (images[:, x:kh+x, y:kw+y] * kernel
                               ).sum(axis=(1, 2))
    return output
