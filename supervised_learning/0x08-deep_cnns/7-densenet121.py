#!/usr/bin/env python3
""" 7. DenseNet-121"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """function that builds the DenseNet-121 architecture as
    described in Densely Connected Convolutional Networks"""
    X = K.Input(shape=(224, 224, 3))
    x = K.layers.BatchNormalization(axis=3)(X)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(2 * growth_rate, (7, 7), (2, 2),
                        kernel_initializer='he_normal', padding='same')(x)
    x = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=2,
                              padding="same")(x)
    x, nb_filters = dense_block(x, 2 * growth_rate, growth_rate, 6)
    x, nb_filters = transition_layer(x, nb_filters, compression)
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 12)
    x, nb_filters = transition_layer(x, nb_filters, compression)
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 24)
    x, nb_filters = transition_layer(x, nb_filters, compression)
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 16)
    x = K.layers.AveragePooling2D(pool_size=(7, 7),
                                  strides=(1, 1))(x)
    x = K.layers.Dense(units=1000, activation='softmax',
                       kernel_initializer='he_normal')(x)
    model = K.models.Model(inputs=X, outputs=x)
    return model
