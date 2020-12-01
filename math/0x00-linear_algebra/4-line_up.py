#!/usr/bin/env python3
def add_arrays(arr1, arr2):
    "function that adds two arrays element-wise"
    arr = []
    if len(arr1) == len(arr2):
        for i in range(len(arr1)):
            arr.append(arr1[i] + arr2[i])
        return arr
    return None