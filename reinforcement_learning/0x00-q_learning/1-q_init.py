#!/usr/bin/env python3
""" Initialize Q-table """
import gym
import numpy as np


def q_init(env):
    """function that initializes the Q-table"""
    action_size = env.action_space.n
    state_size = env.observation_space.n
    Q_table = np.zeros((state_size, action_size))
    return Q_table
