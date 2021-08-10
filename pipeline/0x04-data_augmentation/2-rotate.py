#!/usr/bin/env python3
""" 2. Rotate """

import tensorflow as tf


def rotate_image(image):
    """function that rotates an image by 90 degrees counter-clockwise"""
    return tf.image.rot90(image, k=1, name=None)
