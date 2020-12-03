#!/usr/bin/env python3
""" 15. Slice Like A Ninja  """


def np_slice(matrix, axes={}):
    """ function that slices a matrix along specific axes """
    tmp = []
    for a in range(len(matrix.shape)):
        tmp.append(slice(*axes.get(a, (None, None))))
    return matrix[tuple(tmp)]
