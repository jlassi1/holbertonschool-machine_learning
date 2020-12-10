#!/usr/bin/env python3
""" interation polynomial"""


def poly_integral(poly, C=0):
    """function that calculates the integral of a polynomial"""
    if not isinstance(poly, list) or not all(isinstance(
            x, (int, float)) for x in poly) or poly == []:
        return None
    if not isinstance(C, (int, float)):
        return None
    new_poly = [C]
    for idx, coeff in enumerate(poly):
        if coeff % (idx + 1) == 0:
            new_poly.append(coeff // (idx + 1))
        else:
            new_poly.append(coeff / (idx + 1))
    for i in range(len(new_poly) - 1, 0, -1):
        if new_poly[i] == 0:
            new_poly = new_poly[:-1]
        else:
            break
    return new_poly
