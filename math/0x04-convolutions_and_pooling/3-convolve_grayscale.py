#!/usr/bin/env python3
""" 3. Strided Convolution"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """function that performs a convolution on grayscale images """
    m, h, w = images.shape
    kh, kw = kernel.shape
    s_h = stride[0]
    s_w = stride[1]

    if padding == 'valid':
        final_h = int(((h - kh) / s_h) + 1)
        final_w = int(((w - kw) / s_w) + 1)
        output = np.zeros((m, final_h, final_w))
        image_pad = images

    if padding == "same":
        p_h = int(np.round((kh - 1)/2))
        p_w = int(np.round((kw - 1)/2))

        output = np.zeros((m, h, w))
        image_pad = np.pad(
            array=images,
            pad_width=((0,), (p_h,), (p_w,)),
            mode="constant",
            constant_values=0)

    if isinstance(padding, tuple):
        p_h = padding[0]
        p_w = padding[1]

        final_h = int(((h - kh + 2 * p_h) / s_h) + 1)
        final_w = int(((w - kw + 2 * p_w) / s_w) + 1)

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
