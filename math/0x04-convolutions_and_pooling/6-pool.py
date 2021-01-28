#!/usr/bin/env python3
""" 6. Pooling  """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """function that performs pooling on images"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    s_h, s_w = stride

    final_h = int(np.floor((h - kh) / s_h) + 1)
    final_w = int(np.floor((w - kw) / s_w) + 1)

    output = np.zeros((m, final_h, final_w, c))

    for x in range(final_h):
        for y in range(final_w):
            if mode == "max":
                output[:, x, y, :] = np.max(
                    images[:, x*s_h:kh+x*s_h, y*s_w:kw+y*s_w, :],
                    axis=(1, 2))
            if mode == "avg":
                output[:, x, y, :] = np.average(
                    images[:, x*s_h:kh+x*s_h, y*s_w:kw+y*s_w, :],
                    axis=(1, 2))
    return output
