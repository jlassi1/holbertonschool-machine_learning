#!/usr/bin/env python3
"""10. Hello, sklearn! """
import numpy as np
import sklearn.cluster

# whyyyyy
# def kmeans(X, k):
#     """function that performs K-means on a dataset"""
#     kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
#     return kmeans.cluster_centers_, kmeans.labels_


def kmeans(X, k):
    """function that performs K-means on a dataset"""
    centroids, clss, inertia = sklearn.cluster.k_means(X, k)
    return centroids, clss
