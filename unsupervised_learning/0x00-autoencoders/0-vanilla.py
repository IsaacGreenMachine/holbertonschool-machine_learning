#!/usr/bin/env python3
"""module for autoencoder function"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    creates an autoencoder

    input_dims is an integer containing the dimensions of the model input

    hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively
        the hidden layers should be reversed for the decoder

    latent_dims is an integer containing the dimensions of the latent
    space representation

    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model

    compiled using adam optimization and binary cross-entropy loss
    All layers use relu activation except for last decoder layer,
    which uses sigmoid

    """
    encode_input = keras.Input(input_dims)
    decode_input = keras.Input(latent_dims)

    encoder = create_model(encode_input, hidden_layers, latent_dims, 'relu')

    reverse = hidden_layers.copy()
    reverse.reverse()
    decoder = create_model(decode_input, reverse, input_dims, 'sigmoid')

    autoencoder = keras.Model(encode_input, decoder(encoder(encode_input)))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder


def create_model(input, layers, output_shape, act):
    """
    creates a model from layer info
    """
    lay = keras.layers.Dense(layers[0], activation='relu')(input)
    for layer_size in layers[1:]:
        lay = keras.layers.Dense(layer_size, activation='relu')(lay)
    lay = keras.layers.Dense(output_shape, activation=act)(lay)
    model = keras.Model(input, lay)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model
