#!/usr/bin/env python3
""" Load the Environment """
import gym
import numpy as np


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """function that loads the pre-made FrozenLakeEnv evn from OpenAI’s gym"""
    env = gym.make("FrozenLake8x8-v0",
                   is_slippery=is_slippery,
                   map_name=map_name,
                   desc=desc)
    return env
