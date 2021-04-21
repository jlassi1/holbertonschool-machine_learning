#!/usr/bin/env python3
"""2. Convolutional Autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """function that creates a convolutional autoencoder"""
    input_img = keras.Input(shape=input_dims)

    for i in filters:
        if i == filters[0]:
            encoded = keras.layers.Conv2D(
                i, (3, 3), activation='relu', padding='same')(input_img)
            encoded = keras.layers.MaxPooling2D(
                (2, 2), padding='same')(encoded)
        else:
            encoded = keras.layers.Conv2D(
                i, (3, 3), activation='relu', padding='same')(encoded)
            encoded = keras.layers.MaxPooling2D(
                (2, 2), padding='same')(encoded)

    encoder = keras.Model(input_img, encoded)

    decoder_input = keras.Input(shape=latent_dims)
    decoded = keras.layers.Conv2D(
        filters[-1], (3, 3), activation='relu', padding='same')(decoder_input)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    for i in range(len(filters) - 2, 0, -1):
        decoded = keras.layers.Conv2D(
            filters[i], (3, 3), activation='relu', padding='same')(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    decoded = keras.layers.Conv2D(
        filters[0], (3, 3), padding='valid', activation='relu')(decoded)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    decoded = keras.layers.Conv2D(
        input_dims[-1], (3, 3), activation='sigmoid', padding='same')(decoded)

    decoder = keras.Model(decoder_input, decoded)

    auto = keras.Model(input_img, decoder(encoded))
    auto.compile("Adam", 'binary_crossentropy')

    return encoder, decoder, auto
