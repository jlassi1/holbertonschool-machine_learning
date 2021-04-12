#!/usr/bin/env python3
""" 6. The Baum-Welch Algorithm """
""" http://www.adeveloperdiary.com/data-science/machine-learning/derivation-and-implementation-of-baum-welch-algorithm-for-hidden-markov-model/"""
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
                X = np.sum(F[:, t - 1] * Transition.T, axis=1)
                a = Emission[:, idx_obs]
                F[:, t] = np.multiply(a, X)
        P = np.sum(F[:, T - 1])
        return F
    except Exception:
        return None


def backward(Observation, Emission, Transition, Initial):
    """function that performs the backward algorithm for a HMM"""
    try:
        T = Observation.shape[0]
        N, _ = Emission.shape
        Beta = np.ones((N, T))

        for t in range(T - 2, -1, -1):
            for s in range(N):
                idx_obs = Observation[t + 1]

                Beta[s, t] = np.sum(
                    Beta[:, t + 1] * Emission[:, idx_obs] * Transition[s, :])

        P = np.sum(Beta[:, 0] * Emission[:, Observation[0]] * Initial.T)
        return Beta
    except Exception:
        return None

def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """function that performs the Baum-Welch algorithm for a HMM"""
    T = Observations.shape[0]
    N, M = Emission.shape
    for n in range(iterations):
        alpha = forward(Observations, Emission, Transition, Initial)
        beta = backward(Observations, Emission, Transition, Initial)
 
        xi = np.zeros((M, M, T - 1))
        for i in range(T - 1):
            den = np.dot(np.dot(alpha[:, i].T, Transition) *
                         Emission[:, Observations[i + 1]].T, beta[:, i + 1])
            for j in range(M):
                num = alpha[j, i] * Transition[j] *\
                    Emission[:, Observations[i + 1]].T * beta[:, i + 1].T
                xi[j, :, i] = num / den
 
        gamma = np.sum(xi, axis=1)
        Transition = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
 
        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
 
        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, V == l], axis=1)
 
        Emission = np.divide(b, denominator.reshape((-1, 1)))
 
    return a, b
