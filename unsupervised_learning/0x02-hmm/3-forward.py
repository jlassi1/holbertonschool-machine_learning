#!/usr/bin/env python3
"""3. The Forward Algorithm """
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """function that performs the forward algorithm for a HMM"""
    try:
        T = Observation.shape[0]
        N, _ = Emission.shape
        F = np.empty((N, T))

        for t in range(T):
            idx_obs = Observation[t]
            if t == 0:
                F[:, t] = Initial.T * Emission[:, idx_obs]
            else:
                X = np.sum(F[:, t - 1]*Transition.T, axis=1)

                a = Emission[:, idx_obs]
                F[:, t] = np.multiply(a, X)
        P = np.sum(F[:, T - 1])
        return P, F
    except Exception:
        return None, None
