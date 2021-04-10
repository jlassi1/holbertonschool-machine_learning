#!/usr/bin/env python3
"""4. The Viretbi Algorithm """
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """function that calculates the most likely sequence
    of hidden states for a hidden markov model"""
    try:
        T = Observation.shape[0]
        N, _ = Emission.shape
        viterbi = np.empty((N, T))
        backpointer = np.empty_like(viterbi)
        for t in range(T):
            for j in range(N):
                idx_obs = Observation[t]
                if t == 0:
                    viterbi[:, t] = Initial.T * Emission[:, idx_obs]
                    backpointer[:, 0] = 0
                else:
                    X = viterbi[:, t - 1] * Transition[:, j]
                    a = Emission[j, idx_obs]
                    viterbi[j, t] = np.max(a * X)
                    backpointer[j, t] = np.argmax(
                        viterbi[:, t - 1]*Emission[j, idx_obs]*Transition[:, j]
                        )

        bestpathprob = np.max(viterbi[:, T - 1])
        bestpathpointer = np.argmax(viterbi[:, T - 1]).astype(int)
        # print(bestpathpointer)
        bestpath = [bestpathpointer]
        # Reconstruction:
        for i in range(T - 1, 0, -1):
            bestpathpointer = backpointer[bestpathpointer, i].astype(int)
            # print(bestpathpointer)
            bestpath.append(bestpathpointer)
        return bestpath[::-1], bestpathprob
        # return path, P
    except Exception:
        return None, None
