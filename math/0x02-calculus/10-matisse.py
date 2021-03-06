#!/usr/bin/env python3
"""  derivation polynomial"""


def poly_derivative(poly):
    """function that calculates the derivative of a polynomial"""
    if not isinstance(poly, list) or not all(
            isinstance(x, (int, float)) for x in poly) or poly == []:
        return None
    if len(poly) <= 1:
        return[0]
    new_poly = [coeff * idx for idx, coeff in enumerate(poly[1:], 1)]
    if all(x == 0 for x in new_poly):
        return [0]
    return new_poly
