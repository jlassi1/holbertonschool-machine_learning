#!/usr/bin/env python3
"""  derivation polynomial"""


def poly_derivative(poly):
    """function that calculates the derivative of a polynomial"""
    if not isinstance(poly, list) or not all(isinstance(x, int) for x in poly):
        return None
    if len(poly) <= 1:
        return[0]
    return [coeff * idx for idx, coeff in enumerate(poly[1:], 1)]
