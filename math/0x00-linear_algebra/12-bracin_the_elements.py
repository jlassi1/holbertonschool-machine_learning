#!/usr/bin/env python3
"""12. Bracing The Elements """


def np_elementwise(mat1, mat2):
    """ function that performs element-wise addition, subtraction, multiplication, and division """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return add, sub, mul, div
