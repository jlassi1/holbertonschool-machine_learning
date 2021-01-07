#!/usr/bin/env python3
""" One-Hot Encode """
import numpy as np


def one_hot_encode(Y, classes):
    """function that converts a numeric
    label vector into a one-hot matrix"""
    try:
        b = np.zeros((classes, Y.max()+1))
        b[np.arange(classes), Y] = 1
        return b.T
    except Exception:
        return None
