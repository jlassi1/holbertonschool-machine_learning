#!/usr/bin/env python3
""" 5. LeNet-5 (Keras)"""
import tensorflow.keras as K


def lenet5(X):
    """function that builds a modified version of
    the LeNet-5 architecture using keras"""
    # the he_normal initialization method
    init = K.initializers.he_normal()
    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv1 = K.layers.Conv2D(6, (5, 5),
                            padding='same', kernel_initializer=init,
                            activation='relu')(X)
    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Mpool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv2 = K.layers.Conv2D(16, (5, 5),
                            padding='valid', kernel_initializer=init,
                            activation='relu')(Mpool1)
    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Mpool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    # Fully connected layer with 120 nodes
    FC = K.layers.Flatten()(Mpool2)
    FC1 = K.layers.Dense(120, kernel_initializer=init,
                         activation='relu')(FC)
    # Fully connected layer with 84 nodes
    FC2 = K.layers.Dense(84, kernel_initializer=init,
                         activation='relu')(FC1)
    # Fully connected softmax output layer with 10 nodes
    output = K.layers.Dense(10, kernel_initializer=init,
                            activation='softmax')(FC2)
    # build a neural network with all previous layer
    model = K.models.Model(inputs=X, outputs=output)
    # model compiled to use Adam optimization and accuracy metrics
    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])
    return model
