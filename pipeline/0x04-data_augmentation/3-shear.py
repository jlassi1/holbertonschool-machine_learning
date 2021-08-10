#!/usr/bin/env python3
"""3. Shear"""

import tensorflow as tf


def shear_image(image, intensity):
    """function that randomly shears an image"""
    return tf.keras.preprocessing.image.random_shear(
        image, intensity)
