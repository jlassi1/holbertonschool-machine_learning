#!/usr/bin/env python3
"""0. Flip """

import tensorflow as tf


def flip_image(image):
    """function that flips an image horizontally"""
    return tf.image.random_flip_left_right(image)
