#!/usr/bin/env python3
""" 0. Initialize """
import tensorflow as tf
import numpy as np


class NST:
    """class NST that performs tasks for neural style transfer"""

    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1',
                    'block5_conv1']

    content_layer = 'block5_conv2'
    tf.enable_eager_execution()

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """ initialization of parameters """
        if (not isinstance(style_image, np.ndarray)
                or len(style_image.shape) is not 3
                or style_image.shape[2] is not 3):
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)')
        if (not isinstance(content_image, np.ndarray)
                or len(content_image.shape) is not 3
                or content_image.shape[2] is not 3):
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)')

        if alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if beta < 0:
            raise TypeError('beta must be a non-negative number')
        self.content_image = self.scale_image(content_image)
        self.style_image = self.scale_image(style_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """function that rescales an image such that its pixels values
        are between 0 and 1 and its largest side is 512 pixels"""
        if (not isinstance(image, np.ndarray)
            or len(image.shape) is not 3
                or image.shape[2] is not 3):
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)')
        largers = max(image.shape[0], image.shape[1])
        coeff = 512 / largers
        h_new = int(image.shape[0] * coeff)
        w_new = int(image.shape[1] * coeff)

        image = np.expand_dims(image, axis=0)
        image = tf.image.resize_bicubic(image, (h_new, w_new))

        image = tf.clip_by_value(image / 255, 0, 1)

        return tf.cast(image, tf.float32)
