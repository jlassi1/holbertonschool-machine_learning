#!/usr/bin/env python3
""" Epsilon Greedy"""
import gym
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """function that uses epsilon-greedy to determine the next action"""
    p = np.random.uniform(0, 1)
    if p < epsilon:
        # Exploration: select random action
        j = np.random.randint(Q.shape[1])
        print(Q.shape[1])
    else:
        # Exploitation: select the best known action
        j = np.argmax([a for a in Q[state, :]])
    return j
