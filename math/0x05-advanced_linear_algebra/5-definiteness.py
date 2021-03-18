#!/usr/bin/env python3
""" 5. Definiteness """
import numpy as np


def definiteness(matrix):
    """function that calculates the definiteness of a matrix"""
    definitess = [
        'Positive definite',
        'Positive semi-definite',
        'Negative semi-definite',
        'Negative definite',
        'Indefinite']
    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')
    if matrix.size == 0 or matrix.ndim == 0 or matrix.shape[0] != matrix.shape[1]:
        return None
    try:
        w, v = np.linalg.eig(matrix)
        if all(w > 0):
            return definitess[0]
        if all(w >= 0):
            return definitess[1]
        if all(w < 0):
            return definitess[3]
        if all(w <= 0):
            return definitess[2]
        else:
            return definitess[4]
    except Exception:
        return None
