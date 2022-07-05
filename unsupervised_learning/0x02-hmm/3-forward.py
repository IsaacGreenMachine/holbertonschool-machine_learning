#!/usr/bin/env python3
"""module for forward function"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    performs the forward algorithm for a hidden markov model

    Observation - a numpy.ndarray of shape (T,)
    that contains the index of the observation
        T is the number of observations

    Emission - a numpy.ndarray of shape (N, M) containing
    the emission probability of a specific observation given a hidden state
        Emission[i, j] is the probability of
        observing j given the hidden state i
        N is the number of hidden states
        M is the number of all possible observations

    Transition - a 2D numpy.ndarray of shape (N, N)
    containing the transition probabilities
        Transition[i, j] is the probability of
        transitioning from the hidden state i to j

    Initial - numpy.ndarray of shape (N, 1) containing the
    probability of starting in a particular hidden state

    Returns: P, F, or None, None on failure
        P is the likelihood of the observations given the model
        F is a numpy.ndarray of shape (N, T) containing
        the forward path probabilities
        F[i, j] is the probability of being in hidden
        state i at time j given the previous observations
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

    # initalize alphas using priors
    alphas = (Initial.T * Emission[:, Observation[0]]).T

    n, m = Emission.shape

    # for each observation:
    for tstep in range(1, Observation.shape[0]):
        # prev. alpha value *
        # P(seeing this observation) *
        # P(result given observation)
        rt = (alphas[:, tstep-1].T * Transition.T.reshape(n, 1, n)
              ) * Emission[:, Observation[tstep]].reshape(n, 1, 1)
        # adding prev. alphas sum to new alpha
        alphas = np.concatenate((alphas, rt.sum(-1)), axis=1)

    # sum up all alphas
    # P should be the sum of all alphas(i) where i is the desired outcome
    # F is keeping track of alpha values each step
    return np.sum(alphas[:, -1]), alphas
