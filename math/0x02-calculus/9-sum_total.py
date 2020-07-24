#!/usr/bin/env python3
"""Here we are going to do a sum"""


def summation_i_squared(n):
    """function that return the sum"""
    if not isinstance(n, (int, float)) or n != int(n) or n < 1:
        return None
    return sum(map(lambda i: i*i, range(1, n+1)))
