#!/usr/bin/env python3
""" 5. The Backward Algorithm """
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """function that performs the backward algorithm for a HMM"""
    try:
        T = Observation.shape[0]
        N, _ = Emission.shape
        Beta = np.empty((N, T))
        Beta[:, T - 1] = 1

        for t in range(T - 2, -1, -1):
            for s in range(N):
                idx_obs = Observation[t + 1]

                Beta[s, t] = np.sum(
                    Beta[:, t + 1] * Emission[:, idx_obs] * Transition[s, :])

        P = np.sum(Beta[:, 0] * Emission[:, Observation[0]] * Initial.T)
        return P, Beta
    except Exception:
        return None, None
