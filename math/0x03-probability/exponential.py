#!/usr/bin/env python3
"""Create a class Exponential that represents a poisson distribution"""


def __init__(self, data=None, lambtha=1.):
    """init Class"""
    self.lambtha = float(lambtha)
    if data is None and self.lambtha <= 0:
        raise ValueError('lambtha must be a positive value')
    if data is not None:
        if type(data) != list:
            raise TypeError('data must be a list')
        if len(data) < 2:
            raise ValueError('data must contain multiple values')
        self.lambtha = (sum(data)/len(data))**-1
