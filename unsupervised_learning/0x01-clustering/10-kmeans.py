#!/usr/bin/env python3
"""10. Hello, sklearn! """
import numpy as np
import sklearn.cluster


def kmeans(X, k):
    """function that performs K-means on a dataset"""
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return kmeans.cluster_centers_, kmeans.fit_predict(X)
