#!/usr/bin/env python3
""" 15. Slice Like A Ninja  """


def np_slice(matrix, axes={}):
    """ function that slices a matrix along specific axes """
    slider = []
    for m in range(len(matrix.shape)):
        slider.append(slice(*axes.get(m, None)))
    return matrix[tuple(slider)]
