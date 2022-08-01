#!/usr/bin/env python3
"""module for Generator class"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Generator(nn.Module):
    """
    instantiates a generator model in a GAN

    The network has three layers and two tanh activation
    functions after the first and second layer.

    The layers and activation functions are contained
    inside of a nn.Sequential wrapper class.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """sets instance variables for generator"""
        super().__init__()
        self.model = nn.Sequential(
          nn.Linear(input_size, hidden_size), nn.Tanh(),
          nn.Linear(hidden_size, hidden_size), nn.Tanh(),
          nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        """feed forward function"""
        return self.model(x)
