#!/usr/bin/env python3
"""3. Variational Autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """function that creates a variational autoencoder"""

    input_img = keras.Input(shape=(input_dims,))

    for i in hidden_layers:
        if i is hidden_layers[0]:
            encoded = keras.layers.Dense(i, activation='relu')(input_img)
        else:
            encoded = keras.layers.Dense(i, activation='relu')(encoded)

    # latent = keras.layers.Dense(latent_dims, activation=None)(encoded)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(encoded)
    z_log_sigma = keras.layers.Dense(latent_dims, activation=None)(encoded)

    def sampling(args):
        """sampling a new similar points from the latent space"""
        z_mean, z_log_sigma = args
        epsilon = keras.backend.random_normal(shape=(latent_dims,), mean=0.0,
                                              stddev=1.0)
        return z_mean + keras.backend.exp(z_log_sigma / 2) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(
        latent_dims,))([z_mean, z_log_sigma])
    # z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])

    encoder = keras.Model(input_img, [z_mean, z_log_sigma, z])
    decoder_input = keras.Input(shape=(latent_dims,))

    for i in hidden_layers[::-1]:
        if i is hidden_layers[-1]:
            decoded = keras.layers.Dense(i, activation='relu')(decoder_input)
        else:
            decoded = keras.layers.Dense(i, activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    decoder = keras.Model(decoder_input, decoded)
    output = decoder(encoder(input_img)[2])

    auto = keras.Model(input_img, output)

    def loss_vae(input_img, output):
        reconstruction_loss = keras.losses.binary_crossentropy(
            input_img, output)
        reconstruction_loss *= input_dims
        kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean) \
            - keras.backend.exp(z_log_sigma)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
        return vae_loss
    # auto.add_loss(vae_loss)

    auto.compile(optimizer='adam', loss=loss_vae)

    return encoder, decoder, auto
