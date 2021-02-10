#!/usr/bin/env python3
"""1. Inception Network"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """function that builds the inception network as described
    in Going Deeper with Convolutions (2014)"""
    # create an input model with shape=(224, 224, 3)
    X = K.Input(shape=(224, 224, 3))
    # create an output of the inception block
    conv_7x7 = K.layers.Conv2D(64, (7, 7), strides=(2, 2),
                               padding='same', activation='relu')(X)

    M_pool = K.layers.MaxPool2D((3, 3),
                                strides=(2, 2), padding='same')(conv_7x7)
    Norm = K.layers.BatchNormalization()(M_pool)
    conv_3x3_reduce = K.layers.Conv2D(64, (1, 1),
                                      strides=(1, 1), padding='same',
                                      activation='relu')(Norm)
    conv_3x3 = K.layers.Conv2D(192, (3, 3),
                               strides=(1, 1), padding='same',
                               activation='relu')(conv_3x3_reduce)
    Norm = K.layers.BatchNormalization()(conv_3x3)
    M_pool = K.layers.MaxPool2D((3, 3),
                                strides=(2, 2), padding='same')(Norm)
    a3 = inception_block(M_pool, [64, 96, 128, 16, 32, 32])
    b3 = inception_block(a3, [128, 128, 192, 32, 96, 64])
    M_pool = K.layers.MaxPool2D((3, 3),
                                strides=(2, 2), padding='same')(b3)
    a4 = inception_block(M_pool, [192, 96, 208, 16, 48, 64])

    x0 = K.layers.AveragePooling2D((5, 5), strides=(3, 3))(a4)
    x0 = K.layers.Conv2D(128, (1, 1), strides=(1, 1),
                         padding="same", activation='relu')(x0)
    x0 = K.layers.Flatten()(x0)
    x0 = K.layers.Dense(1024, activation='relu')(x0)
    x0 = K.layers.Dropout(0.7)(x0)
    X0 = K.layers.Dense(1000, activation='softmax')(x0)

    b4 = inception_block(a4, [160, 112, 224, 24, 64, 64])
    c4 = inception_block(b4, [128, 128, 256, 24, 64, 64])
    d4 = inception_block(c4, [112, 144, 288, 32, 64, 64])

    x1 = K.layers.AveragePooling2D((5, 5), strides=(3, 3))(d4)
    x1 = K.layers.Conv2D(128, (1, 1), strides=(1, 1),
                         padding="same", activation='relu')(x1)
    x1 = K.layers.Flatten()(x1)
    x1 = K.layers.Dense(1024, activation='relu')(x1)
    x1 = K.layers.Dropout(0.7)(x1)
    x1 = K.layers.Dense(1000, activation='softmax')(x1)

    e4 = inception_block(d4, [256, 160, 320, 23, 128, 128])
    M_pool = K.layers.MaxPool2D((3, 3),
                                strides=(2, 2), padding='same')(e4)
    a5 = inception_block(M_pool, [256, 160, 320, 32, 128, 128])
    b5 = inception_block(a5, [384, 192, 384, 48, 128, 128])
    AVG_pool = K.layers.AveragePooling2D((7, 7), strides=(1, 1))(b5)
    dropout = K.layers.Dropout(0.4)(AVG_pool)
    linear = K.layers.Flatten()(dropout)
    Y = K.layers.Dense(1000, activation='softmax')(dropout)
    # a keras model frome the input and the outputs after the inception block
    model = K.models.Model(inputs=X, outputs=Y)
    return model
