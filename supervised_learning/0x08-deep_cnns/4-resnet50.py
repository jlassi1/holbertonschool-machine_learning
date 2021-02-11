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
    init = K.initializers.he_normal()
    model50 = K.applications.ResNet50(include_top=False,
                                      input_tensor=X, input_shape=(
                                          224, 224, 3))
    avg = K.layers.AveragePooling2D(pool_size=(7, 7),
                                    strides=(1, 1))(model50.output)
    Y = K.layers.Dense(units=1000, activation='softmax',
                       kernel_initializer=init)(avg)
    model = K.models.Model(inputs=model50.input, outputs=Y)
    return model
