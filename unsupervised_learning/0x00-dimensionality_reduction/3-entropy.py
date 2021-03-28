#!/usr/bin/env python3
""" 3. Entropy """
import numpy as np


def HP(Di, beta):
    """function  that calculates the Shannon entropy
    and P affinities relative to a data point"""
    # original equation of P(ij)
    # pjji=exp  kxi xjk2=2σ2i∑k6=iexp  kxi xkk2=2σ2i;

    Pi = np.exp(-Di) / np.sum(np.exp(-Di))

    # equation of H(i)
    # H(Pi) = ∑jpjjilog2pjj
    Hi = -np.sum(np.dot(Pi, np.log2(Pi)))

    return (Hi, Pi)
