#!/usr/bin/env python3
"""2. Precision """
import numpy as np


def precision(confusion):
    """function that calculates the precision
    for each class in a confusion matrix"""
    """
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    TN = confusion.sum() - (TP+FP+FN)
    """
    """ precision = TP/(TP+FP) = accuracy """
    return confusion.diagonal() / confusion.sum(axis=0)
