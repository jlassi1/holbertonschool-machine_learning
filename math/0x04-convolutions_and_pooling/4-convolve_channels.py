#!/usr/bin/env python3
""" 4. Convolution with Channels """
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """function that performs a convolution on images with channels"""
    m, h, w, c = images.shape
    kh, kw, c = kernel.shape
    s_h, s_w = stride

    if padding == 'valid':
        final_h = int(np.floor(((h - kh)) / s_h + 1))
        final_w = int(np.floor(((w - kw)) / s_w + 1))
        output = np.zeros((m, final_h, final_w, c))
        image_pad = images

    if padding == "same":
        p_h = int(np.ceil(((h - 1) * s_h + kh - h) / 2))
        p_w = int(np.ceil(((w - 1) * s_w + kw - w) / 2))
        final_h = int(np.floor((h - kh + 2 * p_h) / s_h) + 1)
        final_w = int(np.floor((w - kw + 2 * p_w) / s_w) + 1)

        output = np.zeros((m, final_h, final_w, c))
        image_pad = np.pad(
            array=images,
            pad_width=((0,), (p_h,), (p_w,)),
            mode="constant",
            constant_values=0)

    if isinstance(padding, tuple):
        p_h, p_w = padding
        final_h = int(np.floor((h - kh + 2 * p_h) / s_h) + 1)
        final_w = int(np.floor((w - kw + 2 * p_w) / s_w) + 1)

        output = np.zeros((m, final_h, final_w, c))
        image_pad = np.pad(
            array=images,
            pad_width=((0,), (p_h,), (p_w,)),
            mode="constant",
            constant_values=0)

    for x in range(final_h):
        for y in range(final_w):
            output[:, x, y, :] = (
                image_pad[:, x*s_h:kh+x*s_h, y*s_w:kw+y*s_w, :] * kernel
                               ).sum(axis=(1, 2))
    return output.astype(np.uint8)
