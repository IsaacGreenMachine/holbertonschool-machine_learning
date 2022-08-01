#!/usr/bin/env python3
"""module for train_gan function"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def train_gan(
        lr=1e-3, batch_size=512, steps=5000, dis_steps=20,
        gen_steps=20, gen_size=(1, 16, 1), dis_size=(1, 16, 1)):
    """
    trains a GAN

    lr - learning rate used in optimization

    batch_size - batch size used in forward/backward pass

    steps - number of iterations


    dis/gen_steps - number of steps used to train discriminator/generator

    returns : generated distribution from the Generator (torch.Tensor())
    """
    Generator = __import__('0-generator.py').Generator
    Discriminator = __import__('1-discriminator.py').Discriminator
    sample_Z = __import__('2-sample_Z.py').sample_Z
    train_dis = __import__('3-train_discriminator.py').train_dis
    train_gen = __import__('4-train_generator.py').train_gen
    if type(lr) not in [int, float]:
        raise TypeError("learning rate lr must be of type int or float")
    if lr <= 0:
        raise ValueError("learning rate lr must be bigger than 0")
    if type(batch_size) not in [int, float]:
        raise TypeError("batch_size must be of type int or float")
    if batch_size < 1:
        raise ValueError("batch size must be greater than 0")
    if type(steps) not in [int, float]:
        raise TypeError("steps must be of type int or float")
    if steps < 1:
        raise ValueError("steps must be greater than 0")
    if type(dis_steps) not in [int, float]:
        raise TypeError("dis_steps must be of type int or float")
    if dis_steps < 1:
        raise ValueError("dis_steps size must be greater than 0")
    if type(gen_steps) not in [int, float]:
        raise TypeError("gen_steps must be of type int or float")
    if gen_steps < 1:
        raise ValueError("gen_steps must be greater than 0")
    if type(gen_size) not in [tuple, list]:
        raise TypeError("gen_size must be a tuple or list")
    if len(gen_size) != 3:
        raise ValueError("gen_size must be of length 3")
    if type(dis_size) not in [tuple, list]:
        raise TypeError("dis_size must be a tuple or list")
    if len(dis_size) != 3:
        raise ValueError("dis_size must be of length 3")
    if (dis_size[-1] != gen_size[0]):
        raise ValueError(
            """discriminator output (dis_size) and generator
            input (gen_size) must match""")
    if (gen_size[-1] != dis_size[0]):
        raise ValueError(
            """discriminator input (dis_size) and generator
            output (gen_size) must match""")

    generator = Generator(gen_size[0], gen_size[1], gen_size[2])
    discriminator = Discriminator(dis_size[0], dis_size[1], dis_size[2])
    gen_opt = optim.SGD(generator.parameters(), lr)
    dis_opt = optim.SGD(discriminator.parameters(), lr)
    loss = torch.nn.BCELoss()

    for step in range(steps):
        if step % 500 == 0:
            print(
                f"""step {step} : loss
                {train_gen(
                    generator, discriminator, gen_size[0], dis_size[0],
                    batch_size, 1, gen_opt, loss
                    )[0]
                    }""")
        else:
            train_gen(generator, discriminator,
                      gen_size[0], dis_size[0], batch_size, 1, gen_opt, loss)
            train_dis(generator, discriminator,
                      gen_size[0], dis_size[0], batch_size, 1, gen_opt, loss)

    return generator(sample_Z('G', 1, 100))
