#!/usr/bin/env python3
"""0. "Vanilla" Autoencoder """
import numpy as np
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """function that creates a sparse autoencoder"""
    input_img = keras.Input(shape=(input_dims,))

    for i in hidden_layers:
        if i == hidden_layers[0]:
            encoded = keras.layers.Dense(
                i,
                activation='relu',
                activity_regularizer=keras.regularizers.l1(lambtha))(input_img)
        else:
            encoded = keras.layers.Dense(
                i,
                activation='relu',
                activity_regularizer=keras.regularizers.l1(lambtha))(encoded)

    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha))(encoded)

    encoder = keras.Model(input_img, latent)

    decoder_input = keras.Input(shape=(latent_dims,))
    for i in hidden_layers[::-1]:
        if i == hidden_layers[1]:
            decoded = keras.layers.Dense(
                i,
                activation='relu',
                activity_regularizer=keras.regularizers.l1(lambtha)
                )(decoder_input)
        else:
            decoded = keras.layers.Dense(
                i,
                activation='relu',
                activity_regularizer=keras.regularizers.l1(lambtha)
                )(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    decoder = keras.Model(decoder_input, decoded)

    auto = keras.Model(input_img, decoder(latent))
    auto.compile("Adam", 'binary_crossentropy')

    return encoder, decoder, auto
