#!/usr/bin/env python3
""" 1. Same Convolution """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """function that performs a same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    p_h = int(np.round((kh - 1)/2))
    p_w = int(np.round((kw - 1)/2))

    #final_h = h + 2*p_h - kh + 1
    #final_w = w + 2*p_w - kw + 1

    output = np.zeros((m, h, w))
    image_pad = np.pad(
        array=images,
        pad_width=((0,), (p_h,), (p_w,)),
        mode="constant",
        constant_values=0)

    for x in range(h):
        for y in range(w):
            output[:, x, y] = (image_pad[:, x:kh+x, y:kw+y] * kernel
                               ).sum(axis=(1, 2))
    return output
