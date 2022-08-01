#!/usr/bin/env python3
"""module for train_dis function"""
from random import sample
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def train_dis(
        Gen, Dis, dInputSize, gInputSize, mbatchSize, steps, optimizer, crit
        ):
    """
    Gen and Dis are Discriminator and Generator Models.

    dInputSize is the input size of Discriminator input data.

    gInputSize is the input size of Generator input data.

    mbatchSize is the batch size for training.

    steps is number of steps for training.

    optimizer is a PyTorch stochastic gradient descent (SGD) optimizer object.

    crit is a PyTorch BCEloss function.

    returns the loss of the fake data, real data, and dataset torch.Tensor()
    """
    sample_Z = __import__('2-sample_Z.py').sample_Z
    for step in range(steps):
        # getting sample from standard normal distribution
        # (mu = 0, std.dev = 1) for discriminator
        realData = sample_Z("D", dInputSize, mbatchSize, 0, 1)
        # generating labels for sample.
        # (all are from real data, so all labels are '1')
        realDataLabels = torch.ones((mbatchSize, 1))

        # getting data from generator (random)
        generatorData = sample_Z("G", gInputSize, mbatchSize)
        genOutput = Gen(generatorData)
        genOutputLabels = torch.zeros((mbatchSize, 1))

        # concatenating all valid and generator data/labels
        # to pass through discriminator
        all_data = torch.cat((realData, genOutput))
        all_labels = torch.cat((realDataLabels, genOutputLabels))

        # train discriminator
        # zeroing out gradients instead of summing each time
        Dis.zero_grad()
        # getting output from discriminator of all valid and generator data
        output = Dis(all_data)
        # calculating loss based on output of discriminator
        # using labels as reference
        loss = crit(output, all_labels)
        # backprop
        loss.backward()
        # perform single optimization step
        optimizer.step()
    real_output = Dis(realData)
    real_loss = crit(real_output, realDataLabels)
    generator_out = Dis(genOutput)
    generator_loss = crit(generator_out, genOutputLabels)
    return generator_loss, real_loss, all_data
