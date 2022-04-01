#!/usr/bin/env python3
"""module for save_config and load_config"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    saves a model’s configuration in JSON format

    network - model whose configuration should be saved
    filename - path of the file that the configuration should be saved to
    Returns: None
    """
    with open(filename, "w") as f:
        f.write(network.to_json())
    return None


def load_config(filename):
    """
    loads a model with a specific configuration

    filename - path of JSON file with model configuration
    Returns: the loaded model
    """
    with open(filename, "r") as f:
        model = f.read()
    return K.models.model_from_json(model)
