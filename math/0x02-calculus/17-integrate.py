#!/usr/bin/env python3
""" interation polynomial"""


def poly_integral(poly, C=0):
    """function that calculates the integral of a polynomial"""
    if not isinstance(poly, list) or not isinstance(C, (
            int, float)) or not all(isinstance(x, (int, float)) for x in poly):

        return None
    new_poly = [C]
    for idx, coeff in enumerate(poly):
        if coeff % (idx + 1) == 0:
            new_poly.append(coeff // (idx + 1))
        else:
            new_poly.append(coeff / (idx + 1))
    return new_poly
