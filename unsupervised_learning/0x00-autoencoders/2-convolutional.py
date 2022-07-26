#!/usr/bin/env python3
"""module for autoencoder function"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    creates a convolutional autoencoder

    input_dims is a tuple of integers containing the dimensions of
    the model input

    filters is a list containing the number of filters for each
    convolutional layer in the encoder, respectively
      the filters are reversed for the decoder

    latent_dims is a tuple of integers containing the dimensions
    of the latent space representation

    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model

    compiled using adam optimization and binary cross-entropy loss
    All layers use relu activation except for last decoder layer,
    which uses sigmoid
    """
    # encoder
    encode_input = keras.Input(shape=input_dims)
    encode = keras.layers.Conv2D(
        filters[0], 3, padding='same', activation='relu')(encode_input)
    encode = keras.layers.MaxPool2D(padding='same')(encode)
    for filter in filters[1:]:
        encode = keras.layers.Conv2D(
            filter, 3, padding='same', activation='relu')(encode)
        encode = keras.layers.MaxPool2D(padding='same')(encode)
    encoder = keras.Model(encode_input, encode)
    encoder.compile(optimizer='adam', loss='binary_crossentropy')

    # decoder
    decode_input = keras.Input(shape=latent_dims)
    decode = keras.layers.Conv2D(
        filters[-1], 3, padding='same', activation='relu')(decode_input)
    decode = keras.layers.UpSampling2D()(decode)
    for filter in filters[:1:-1]:
        decode = keras.layers.Conv2D(
            filter, 3, padding='same', activation='relu')(decode)
        decode = keras.layers.UpSampling2D()(decode)
    # The second to last convolution should instead use valid padding
    decode = keras.layers.Conv2D(
        filters[0], 3, padding='valid', activation='relu')(decode)
    decode = keras.layers.UpSampling2D()(decode)
    # The last convolution should have the same number of filters as the number
    # of channels in input_dims with sigmoid activation and no upsampling
    decode = keras.layers.Conv2D(
        input_dims[-1], 3, padding='same', activation='sigmoid')(decode)
    decoder = keras.Model(decode_input, decode)
    decoder.compile(optimizer='adam', loss='binary_crossentropy')
    # autoencoder
    autoencoder = keras.Model(encode_input, decoder(encoder(encode_input)))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
