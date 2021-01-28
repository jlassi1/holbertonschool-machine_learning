#!/usr/bin/env python3
""" 2. Convolution with Padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """function that performs a convolution
    on grayscale images with custom padding"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    p_h = padding[0]
    p_w = padding[1]

    final_h = h + 2*p_h - kh + 1
    final_w = w + 2*p_w - kw + 1

    output = np.zeros((m, final_h, final_w))
    image_pad = np.pad(
        array=images,
        pad_width=((0,), (p_h,), (p_w,)),
        mode="constant",
        constant_values=0)

    for x in range(final_h):
        for y in range(final_w):
            output[:, x, y] = (image_pad[:, x:kh+x, y:kw+y] * kernel
                               ).sum(axis=(1, 2))
    return output
