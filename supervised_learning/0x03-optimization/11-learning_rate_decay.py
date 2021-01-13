#!/usr/bin/env python3
"""11. Learning Rate Decay  """
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ function that updates the learning rate using
    inverse time decay in numpy"""
    """ decayed_learning_rate = learning_rate /
    (1 + decay_rate * floor(global_step / decay_step)))"""
    lratr = alpha / (1 + (decay_rate * np.floor(global_step / decay_step)))
    return lratr
