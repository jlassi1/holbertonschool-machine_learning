#!/usr/bin/env python3
""" One-Hot Decode """
import numpy as np


def one_hot_decode(one_hot):
    """function that converts a one-hot matrix into a vector of labels"""
    try:
        if len(one_hot.shape) != 2:
            return None
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
