#!/usr/bin/env python3


def summation_i_squared(n):
    """function that return the sum"""
    sum = 0
    for i in range(n):
        sum += (i + 1)**2
    return sum
