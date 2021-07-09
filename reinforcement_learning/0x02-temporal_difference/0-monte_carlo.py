#!/usr/bin/env python3
""" MC method"""
import numpy as np


def monte_carlo(
        env,
        V,
        policy,
        episodes=5000,
        max_steps=100,
        alpha=0.1,
        gamma=0.99):
    """function that performs the Monte Carlo algorithm"""
    s = V.shape[0]
    n = env.observation_space.n
    discounts = np.array([gamma ** i for i in range(max_steps)])
    for _ in range(episodes):
        episode = []
        state = env.reset()
        for _ in range(max_steps):
            action = policy(state)
            new_state, reward, done, info = env.step(action)
            episode.append([state, reward])
            if done:
                break
            state = new_state

        np_ep = np.array(episode, dtype=int)
        G = 0
        for i, step in enumerate(np_ep[::-1]):
            s, r,  = step
            G = gamma * G + r
            if s not in np_ep[:i, 0]:
                V[s] = V[s] + alpha * (G - V[s])
    return V
