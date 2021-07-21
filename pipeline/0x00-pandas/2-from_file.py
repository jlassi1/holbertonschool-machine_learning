#!/usr/bin/env python3
"""From Numpy """

import numpy as np
import pandas as pd


def from_file(filename, delimiter):
    """function that loads data from a file as a pd.DataFrame"""
    return pd.read_csv(filename, delimiter=delimiter)
