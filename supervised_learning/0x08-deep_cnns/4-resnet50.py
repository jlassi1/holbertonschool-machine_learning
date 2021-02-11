#!/usr/bin/env python3
"""4. ResNet-50 """
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """function that builds the ResNet-50 architecture as described
    in Deep Residual Learning for Image Recognition (2015)"""
    # create an input model with shape=(224, 224, 3)
    X = K.Input(shape=(224, 224, 3))
    # init = K.initializers.he_normal()
    # Z = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
    #                          padding='same',
    #                          strides=(2, 2),
    #                          kernel_initializer=he_init)(X)
    # Z = K.layers.BatchNormalization(axis=3)(Z)
    # Z = K.layers.Activation('relu')(Z)
    # Z = K.layers.MaxPooling2D(pool_size=(3, 3),
    #                                strides=(2, 2),
    #                                padding="same")(Z)

    # Z = projection_block(Z, [64, 64, 256], 1)
    # Z = identity_block(Z, [64, 64, 256])
    # Z = identity_block(Z, [64, 64, 256])
    return K.applications.ResNet50(weights=None,
                                   input_tensor=X, input_shape=(
                                       224, 224, 3), pooling='max')
