#!/usr/bin/env python3
""" play FrozenLake """
import gym
import numpy as np


def play(env, Q, max_steps=100):
    """function that has the trained agent play an episode"""
    env.reset()
    state = 0
    env.render()
    for _ in range(max_steps):
        action = np.argmax(Q[state, :])
        state, reward, done, info = env.step(action)
        env.render()
        if done:
            break
    return reward
