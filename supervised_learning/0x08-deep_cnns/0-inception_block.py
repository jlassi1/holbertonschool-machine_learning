#!/usr/bin/env python3
""" 0-inception block"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """function that builds an inception block as described
    in Going Deeper with Convolutions (2014)"""
    F1, F3R, F3, F5R, F5, FPP = filters
    # create a conv2D with 1*1 conv and F1 the number of his filters
    conv_1x1 = K.layers.Conv2D(F1, (1, 1),
                               padding='same', activation='relu')(A_prev)
    # the 1*1 conv to reduce the 3*3 conv and F3R the number of his filters
    conv_3x3 = K.layers.Conv2D(F3R, (1, 1),
                               padding='same', activation='relu')(A_prev)
    # the 3*3 conv from the output of 1*1 conv and F3 the number of filters
    conv_3x3 = K.layers.Conv2D(F3, (3, 3),
                               padding='same', activation='relu')(conv_3x3)
    # the 1*1 conv to reduce the 5*5 conv and F5R the number of his filters
    conv_5x5 = K.layers.Conv2D(F5R, (1, 1),
                               padding='same', activation='relu')(A_prev)
    # the 5*5 conv from the output of 1*1 conv and F5 the number of filters
    conv_5x5 = K.layers.Conv2D(F5, (5, 5),
                               padding='same', activation='relu')(conv_5x5)
    # Max pooling layer shape 3x3 with same padding to conserve size of output
    pool_proj = K.layers.MaxPool2D((3, 3),
                                   strides=(1, 1), padding='same')(A_prev)
    # the 1*1 conv of the max pooling layer with filtres FPP
    pool_proj = K.layers.Conv2D(FPP, (1, 1),
                                padding='same', activation='relu')(pool_proj)
    # concatenate all the conv layers create
    output = K.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj],
                                  axis=3)
    return output
