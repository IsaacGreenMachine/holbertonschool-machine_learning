#!/usr/bin/env python3
"""module for Discriminator class"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Discriminator(nn.Module):
    """
    instantiates a Discriminator model in a GAN

    The network has three layers and a sigmoid
    activation functions after each layer.

    The layers and activation functions are
    contained inside of a nn.Sequential class.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size), nn.Sigmoid(),
            nn.Linear(hidden_size, output_size), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
