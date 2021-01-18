#!/usr/bin/env python3
"""3. Specificity"""
import numpy as np


def specificity(confusion):
    """function that calculates the specificity for
    each class in a confusion matrix"""
    """ specificity = TN/(TN+FP)"""

    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    TN = confusion.sum() - (TP+FP+FN)

    return TN / (TN + FP)
