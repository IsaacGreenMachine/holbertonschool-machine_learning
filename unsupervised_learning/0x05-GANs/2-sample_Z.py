#!/usr/bin/env python3
"""module for sample_Z function"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def sample_Z(sampleType, InputSize, mbatchSize, mu=None, sigma=None):
    """
    creates input for the generator and discriminator

    mu - the mean of the distribution

    sigma - the standard deviation of the distribution

    sampleType - a variable that selects which model to sample for.
        The variable should accept a "G" or "D" as string values.

    if 'D', output is from a normal distribution for discriminator training
    if 'G', output is random noise for generator training

    returns a torch.Tensor if sampleType is valid, 0 otherwise.

    modified function to actually work instead of broken one from project
    (takes in discriminator/generator input and
    batch size to create the right size sample)
    """
    if sampleType == "D":
        return torch.normal(mean=mu, std=sigma, size=(mbatchSize, InputSize))
    elif sampleType == "G":
        return torch.randn(mbatchSize, InputSize)
    else:
        return 0
