#!/usr/bin/env python3
"""1. Inception Network"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """function that builds the inception network as described
    in Going Deeper with Convolutions (2014)"""
    # create an input model with shape=(224, 224, 3)
    X = K.Input(shape=(224, 224, 3))
    # create an output of the inception network
    Z = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2,
                        padding='same', activation='relu')(X)

    Z = K.layers.MaxPool2D(pool_size=(3, 3),
                           strides=2, padding='same')(Z)
    Z = K.layers.Conv2D(filters=64, kernel_size=(1, 1),
                        strides=1, padding='same',
                        activation='relu')(Z)
    Z = K.layers.Conv2D(
        filters=192,
        kernel_size=(
            3,
            3),
        strides=1,
        padding='same',
        activation='relu')(Z)
    Z = K.layers.MaxPool2D(pool_size=(3, 3),
                           strides=2, padding='same')(Z)
    Z = inception_block(Z, [64, 96, 128, 16, 32, 32])
    Z = inception_block(Z, [128, 128, 192, 32, 96, 64])
    Z = K.layers.MaxPool2D(pool_size=(3, 3),
                           strides=2, padding='same')(Z)

    Z = inception_block(Z, [192, 96, 208, 16, 48, 64])
    Z = inception_block(Z, [160, 112, 224, 24, 64, 64])
    Z = inception_block(Z, [128, 128, 256, 24, 64, 64])
    Z = inception_block(Z, [112, 144, 288, 32, 64, 64])
    Z = inception_block(Z, [256, 160, 320, 32, 128, 128])

    Z = K.layers.MaxPool2D(pool_size=(3, 3),
                           strides=(2, 2), padding='same')(Z)
    Z = inception_block(Z, [256, 160, 320, 32, 128, 128])
    Z = inception_block(Z, [384, 192, 384, 48, 128, 128])
    Z = K.layers.AveragePooling2D(pool_size=(7, 7), strides=1,
                                  padding='valid')(Z)

    Z = K.layers.Dropout(0.4)(Z)
    Y = K.layers.Dense(units=1000, activation='softmax')(Z)
    # a keras model frome the input and the outputs after the inception block
    model = K.models.Model(inputs=X, outputs=Y)
    return model
