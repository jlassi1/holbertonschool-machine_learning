#!/usr/bin/env python3
""" 3. Entropy """
import numpy as np


def HP(Di, beta):
    """function  that calculates the Shannon entropy
    and P affinities relative to a data point"""
    # original equation of P(ij)
    Pi = np.exp(-Di*beta) / np.sum(np.exp(-Di*beta))
    # equation of H(i)
    Hi = -np.sum(np.dot(Pi, np.log2(Pi)))

    return (Hi, Pi)
