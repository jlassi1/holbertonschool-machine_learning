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
    layer = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                            padding='same',
                            strides=2,
                            kernel_initializer="he_normal")(X)
    layer = K.layers.BatchNormalization(axis=3)(layer)
    layer = K.layers.Activation('relu')(layer)
    layer = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="same")(layer)
    layer = projection_block(layer, [64, 64, 256], 1)
    layer = identity_block(layer, [64, 64, 256])
    layer = identity_block(layer, [64, 64, 256])
    layer = projection_block(layer, [128, 128, 512])
    layer = identity_block(layer, [128, 128, 512])
    layer = identity_block(layer, [128, 128, 512])
    iden5 = identity_block(layer, [128, 128, 512])
    layer = projection_block(iden5, [256, 256, 1024])
    layer = identity_block(layer, [256, 256, 1024])
    layer = identity_block(layer, [256, 256, 1024])
    layer = identity_block(layer, [256, 256, 1024])
    layer = identity_block(layer, [256, 256, 1024])
    layer = identity_block(layer, [256, 256, 1024])
    layer = projection_block(layer, [512, 512, 2048])
    layer = identity_block(layer, [512, 512, 2048])
    layer = identity_block(layer, [512, 512, 2048])
    layer = K.layers.AveragePooling2D(pool_size=(7, 7),
                                      strides=(1, 1))(layer)
    Y = K.layers.Dense(units=1000, activation='softmax',
                       kernel_initializer="he_normal")(layer)
    model = K.models.Model(inputs=X, outputs=Y)
    return model

    # create an input model with shape=(224, 224, 3)
    # X = K.Input(shape=(224, 224, 3))
    # init = K.initializers.he_normal()
    # model50 = K.applications.ResNet50(include_top=False,
    #                                   input_tensor=X, input_shape=(
    #                                       224, 224, 3),
    #                                   weights=None)
    # avg = K.layers.AveragePooling2D(pool_size=(7, 7),
    #                                 strides=(1, 1))(model50.output)
    # Y = K.layers.Dense(units=1000, activation='softmax',
    #                    kernel_initializer=init)(avg)
    # model = K.models.Model(inputs=model50.input, outputs=Y)
    # return model
