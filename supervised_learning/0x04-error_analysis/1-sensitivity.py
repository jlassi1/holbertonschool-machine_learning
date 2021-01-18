#!/usr/bin/env python3
"""1. Sensitivity """
import numpy as np


def sensitivity(confusion):
    """function that calculates the sensitivity for
    each class in a confusion matrix"""
    """
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    TN = confusion.sum() - (TP+FP+FN)
    """
    """ recall = TP/(TP+FN) = sensitivity"""
    return confusion.diagonal() / confusion.sum(axis=1)
