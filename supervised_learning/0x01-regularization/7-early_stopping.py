#!/usr/bin/env python3
"""module for early_stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    determines if you should stop gradient descent early

    cost - current validation cost of the neural network
    opt_cost - lowest recorded validation cost of the neural network
    threshold - threshold used for early stopping
    patience - patience count used for early stopping
    count - count of how long the threshold has not been met

    Early stopping should occur when the validation cost of the network
    has not decreased relative to the optimal validation cost by more
    than the threshold over a specific patience count

    Returns: boolean of whether network should be stopped early, updated count
    """
    if cost < opt_cost - threshold:
        return False, 0
    elif count+1 >= patience:
        return True, count+1
    else:
        return False, count+1
