#!/usr/bin/env python3
""" 5. Hue
https://github.com/koshian2/PCAColorAugmentation
https://aparico.github.io/
https://deshanadesai.github.io/notes/Fancy-PCA-with-Scikit-Image?fbclid=IwAR2jlOGvhRfikau1RN9-QIpZdFfLn_PF3y1rfEHDJd0YMGlTxBsyqrcpCJQ
"""
import numpy as np


def pca_color(image, alphas):
    """function that performs PCA color
    augmentation as described in the AlexNet paper"""
    # tf.experimental.numpy.experimental_enable_numpy_behavior()
    # img = image.reshape(-1, 3).astype(np.float32)
    # scaling_factor = np.sqrt(3.0 / np.sum(np.var(img, axis=0)))
    # img *= scaling_factor

    # cov = np.cov(img, rowvar=False)
    # U, S, V = np.linalg.svd(cov)

    # # rand = np.random.randn(3) * 0.1 = alphas
    # delta = np.dot(U, alphas*S)
    # delta = (delta * 255.0).astype(np.int32)[np.newaxis, np.newaxis, :]

    # img_out = np.clip(image + delta, 0, 255).astype(np.uint8)

    # return img_out
    ##########################################################################
    image = image.numpy()
    res = image.reshape(-1, 3)
    m = res.mean(axis=0)
    res = res - m
    R = np.cov(res, rowvar=False)
    evals, evecs = np.linalg.eigh(R)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]

    evals = evals[idx]

    evecs = evecs[:, :3]

    m = np.dot(evecs.T, res.T).T
    print(alphas)

    def data_aug(image=image, alphas=alphas):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):

                for k in range(image.shape[2]):
                    image[i, j, k] = image[i, j, k].astype(
                        np.float32) + alphas[k].astype(np.float32)
    image = image / 255.0
    data_aug(image.astype(np.uint8))
    return image
