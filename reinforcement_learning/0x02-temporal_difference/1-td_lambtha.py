#!/usr/bin/env python3
"""
TD(λ) methode
"""
import numpy as np


def td_lambtha(
        env,
        V,
        policy,
        lambtha,
        episodes=5000,
        max_steps=100,
        alpha=0.1,
        gamma=0.99):
    """function that performs the TD(λ) algorithm"""
    state_size = env.observation_space.n
    e = np.zeros(state_size)
    for i in range(episodes):
        state = env.reset()
        for j in range(max_steps):
            action = policy(state)
            new_state, reward, done, _ = env.step(action)
            delta = reward + gamma * V[new_state] - V[state]
            e[state] += 1.0
            V += alpha * delta * e
            e *= lambtha * gamma
            if done:
                break
            state = new_state
    return V
