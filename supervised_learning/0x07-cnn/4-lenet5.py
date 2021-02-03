#!/usr/bin/env python3
""" 4. LeNet-5 (Tensorflow) """
import tensorflow as tf


def lenet5(x, y):
    """function that builds a modified version of
    the LeNet-5 architecture using tensorflow"""
    init = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_IN")
    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv1 = tf.layers.Conv2D(6, (5, 5),
                             padding='same', kernel_initializer=init,
                             activation=tf.nn.relu)(x)
    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Mpool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv2 = tf.layers.Conv2D(16, (5, 5),
                             padding='valid', kernel_initializer=init,
                             activation=tf.nn.relu)(Mpool1)
    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Mpool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    # Fully connected layer with 120 nodes
    FC = tf.layers.Flatten()(Mpool2)
    FC1 = tf.layers.Dense(120, kernel_initializer=init,
                          activation=tf.nn.relu)(FC)
    # Fully connected layer with 84 nodes
    FC2 = tf.layers.Dense(84, kernel_initializer=init,
                          activation=tf.nn.relu)(FC1)
    # Fully connected softmax output layer with 10 nodes
    y_pred = tf.layers.Dense(10, activation=tf.nn.softmax,
                             kernel_initializer=init)(FC2)

    # Define a loss function
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    # training operation that utilizes Adam optimization
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # predicted accuracy
    correct_prediction = tf.equal(
        tf.argmax(y, axis=1),
        tf.argmax(y_pred, axis=1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return y_pred, train_op, loss, acc
