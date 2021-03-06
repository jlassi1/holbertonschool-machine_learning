#!/usr/bin/env python3
""" One-Hot Encode """
import numpy as np


def one_hot_encode(Y, classes):
    """function that converts a numeric
    label vector into a one-hot matrix"""
    try:
        return np.squeeze(np.eye(classes)[Y.reshape(-1)]).T
    except Exception:
        return None
