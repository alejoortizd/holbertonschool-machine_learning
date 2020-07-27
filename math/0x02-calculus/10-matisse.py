#!/usr/bin/env python3
"""function that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    try:
        iter(poly)
    except TypeError:
        return None
    if poly == [] or any(not isinstance(expo, (int, float)) for expo in poly):
        return None
    if len(poly) == 1:
        return [0]
    return [i*expo for i, expo in enumerate(poly)][1:]
