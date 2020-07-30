#!/usr/bin/env python3
"""Create a class Poisson that represents a poisson distribution"""


class Poisson:
    """Class Poisson"""
    def __init__(self, data=None, lambtha=1.):
        """init class"""
        self.lambtha = float(lambtha)
        if data is None and self.lambtha <= 0:
            raise ValueError('lambtha must be a positive value')
        if data is not None:
            if type(data) != list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """calculate pmf"""
        e = 2.7182818285
        k = int(k)
        if k < 0:
            return 0
        factor = 1
        for i in range(1, k + 1):
            factor *= i
        return(self.lambtha**k * e**-self.lambtha) / factor
