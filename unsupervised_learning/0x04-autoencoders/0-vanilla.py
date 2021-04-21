#!/usr/bin/env python3
"""0. "Vanilla" Autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """function that creates an autoencoder"""

    input_img = keras.Input(shape=(input_dims,))

    for i in hidden_layers:
        if i == hidden_layers[0]:
            encoded = keras.layers.Dense(i, activation='relu')(input_img)
        else:
            encoded = keras.layers.Dense(i, activation='relu')(encoded)

    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)

    encoder = keras.Model(input_img, latent)

    decoder_input = keras.Input(shape=(latent_dims,))
    # print(hidden_layers[::-1])
    for i in hidden_layers[::-1]:
        if i == hidden_layers[-1]:
            decoded = keras.layers.Dense(i, activation='relu')(decoder_input)
        else:
            decoded = keras.layers.Dense(i, activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    decoder = keras.Model(decoder_input, decoded)

    auto = keras.Model(input_img, decoder(encoder(input_img)))
    auto.compile(loss='binary_crossentropy', optimizer='adam')

    return encoder, decoder, auto
