#!/usr/bin/env python3
""" 5. Multiple Kernels """
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """function that performs a convolution on images using multiple kernel"""
    m, h, w, c = images.shape
    kh, kw, c, nc = kernels.shape
    s_h, s_w = stride

    if padding == 'valid':
        final_h = int(np.floor(((h - kh)) / s_h + 1))
        final_w = int(np.floor(((w - kw)) / s_w + 1))
        output = np.zeros((m, final_h, final_w, nc))
        image_pad = images.copy()

    if padding == "same":
        p_h = int(np.ceil(((h - 1) * s_h + kh - h) / 2))
        p_w = int(np.ceil(((w - 1) * s_w + kw - w) / 2))
        final_h = int(np.floor((h - kh + 2 * p_h) / s_h) + 1)
        final_w = int(np.floor((w - kw + 2 * p_w) / s_w) + 1)

        output = np.zeros((m, final_h, final_w, nc))
        image_pad = np.pad(
            array=images,
            pad_width=((0,), (p_h,), (p_w,), (0,)),
            mode="constant",
            constant_values=0)

    if isinstance(padding, tuple):
        p_h, p_w = padding
        final_h = int(np.floor((h - kh + 2 * p_h) / s_h) + 1)
        final_w = int(np.floor((w - kw + 2 * p_w) / s_w) + 1)

        output = np.zeros((m, final_h, final_w, nc))
        image_pad = np.pad(
            array=images,
            pad_width=((0,), (p_h,), (p_w,), (0,)),
            mode="constant",
            constant_values=0)

    for x in range(final_h):
        for y in range(final_w):
            for c in range(nc):
                output[:, x, y, c] = (
                    image_pad[:, x*s_h:kh+x*s_h, y*s_w:kw+y*s_w, :]*kernels[
                        :, :, :, c]).sum(axis=(1, 2, 3))
    return output
