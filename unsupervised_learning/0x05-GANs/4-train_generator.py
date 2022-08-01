#!/usr/bin/env python3
"""module for train_gen function"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def train_gen(
        Gen, Dis, gInputSize, dInputSize, mbatchSize, steps, optimizer, crit):
    """
    The Gen, and Dis are the Discriminator and Generator Objects.

    gInputSize is the input size of Generator input data.

    mbatchSize is the batch size for training.

    steps is the number of steps for training.

    optimizer is a PyTorch stochastic gradient descent (SGD) optimizer object.

    The crit should be a BCEloss method.
    Only random noise should be used for sampling
    The 4 moments should be used in processing the sample.
    returns the error of the generated data, and the dataset torch.Tensor()
    """
    sample_Z = __import__('2-sample_Z.py').sample_Z

    for step in range(steps):
        # generating labels for sample.
        # (trying to get 100% from discriminator, so all are '1')
        realDataLabels = torch.ones((mbatchSize, 1))

        # getting data from generator (random)
        generatorData = sample_Z("G", gInputSize, mbatchSize)
        genOutput = Gen(generatorData)

        # train generator
        # zeroing out gradients instead of summing each time
        Gen.zero_grad()
        # getting output from discriminator of all generator data
        output = Dis(genOutput)
        # calculating loss based on output of discriminator
        # using correct labels as reference
        loss = crit(output, realDataLabels)
        # backprop
        loss.backward()
        # perform single optimization step
        optimizer.step()
    return loss, generatorData
