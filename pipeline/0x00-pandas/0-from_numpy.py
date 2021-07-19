#!/usr/bin/env python3
"""From Numpy """

import numpy as np
import pandas as pd


def from_numpy(array):
    """function that creates a pd.DataFrame from a np.ndarray"""
    alpha = list(map(chr, range(65, 91)))
    x = array.shape[1]
    return pd.DataFrame(array, columns=alpha[0:x])
