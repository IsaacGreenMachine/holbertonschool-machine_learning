#!/usr/bin/env python3
"""module for viterbi function"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    that calculates the most likely sequence of hidden states for a hidden
    markov model:

    Observation is a numpy.ndarray of shape (T,) that contains the index
    of the observation

    T is the number of observations

    Emission is a numpy.ndarray of shape (N, M) containing the
        emission probability of a specific observation given a hidden state
        Emission[i, j] is the probability of observing j
            given the hidden state i
        N is the number of hidden states
        M is the number of all possible observations

    Transition is a 2D numpy.ndarray of shape (N, N) containing the
        transition probabilities
        Transition[i, j] is the probability of transitioning from the
        hidden state i to j

    Initial a numpy.ndarray of shape (N, 1) containing the probability
        of starting in a particular hidden state

    Returns: path, P, or None, None on failure

    path is the a list of length T containing
        the most likely sequence of hidden states

    P is the probability of obtaining the path sequence
    """
    if type(Observation) is not np.ndarray or Observation.ndim != 1:
        return None, None
    if Observation.shape[0] == 0:
        return None, None
    if type(Emission) is not np.ndarray or Emission.ndim != 2:
        return None, None
    if type(Transition) is not np.ndarray or Transition.ndim != 2:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial) != Transition.shape[0]:
        return None, None
    N, _ = Emission.shape
    T = Observation.size

    # initial gamma step
    seq_probs = Initial * Emission[:, Observation[0]][..., np.newaxis]
    buff = np.zeros((N, T))

    # for every observation
    for t in range(1, T):
        # P(observing t) * P() * P
        mat = (Emission[:, Observation[t]] * Transition.reshape(N, 1, N))
        mat = (mat.reshape(N, N) * seq_probs[:, t-1].reshape(N, 1))

        mx = np.max(mat, axis=0).reshape(N, 1)
        seq_probs = np.concatenate((seq_probs, mx), axis=1)
        # psi temporary holding spot
        buff[:, t] = np.argmax(mat, axis=0).T

    # highest probability of all
    P = np.max(seq_probs[:, T-1])
    link = np.argmax(seq_probs[:, T-1])
    path = [link]

    # psi (list of max probs each step) creation
    for t in range(T - 1, 0, -1):
        idx = int(buff[link, t])
        path.append(idx)
        link = idx

    return path[::-1], P

    '''
    out:
      P: chance that this output happens
      4.701733355108224e-252
      F: highest chance of what the weather was each day to
        match clothes in Observations
      (cold(2), cold(2), cloudy(1), snowing(0), snowing(0), ...)
      [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 1, ... ]
      '''
