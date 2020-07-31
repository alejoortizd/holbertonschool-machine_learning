#!/usr/bin/env python3
"""Represent binomial distribution"""


def n_choose_k(n, k):
    """Calculate the number of k combinations"""
    factor_minus_k = 1
    for i in range(k+1, n+1):
        factor_minus_k *= i
    factor_minus_n = 1
    for i in range(1, n - k + 1):
        factor_minus_n *= i
    return factor_minus_k // factor_minus_n


class Binomial:
    """Class Binomial"""
    def __init__(self, data=None, n=1, p=0.5):
        """Initialize binomial distribution"""
        if data is None:
            try:
                self.n = int(n)
                assert n > 0
            except (TypeError, AssertionError):
                raise ValueError('n must be a positive value')
            try:
                self.p = float(p)
                assert 0 < self.p < 1
            except (TypeError, AssertionError):
                raise ValueError('p must be greater than 0 and less than 1')
        elif type(data) != list:
            raise TypeError('data must be a list')
        elif len(data) < 2:
            raise ValueError('data must contain multiple values')
        else:
            mean = sum(data)/len(data)
            var = sum((d - mean)**2 for d in data)/len(data)
            p = 1.0 - var/mean
            n = sum(d/p for d in data)/len(data)
            if n % 1 < 0.5:
                n = int(n)
            else:
                n = int(n + 1)
            p = sum(d/n for d in data)/len(data)
            self.n = n
            self.p = float(p)

    def pmf(self, k):
        """Calculate pmf"""
        try:
            k = int(k)
            assert 0 <= k <= self.n
        except (TypeError, AssertionError):
            return 0
        return n_choose_k(self.n, k) * self.p**k * (1-self.p)**(self.n-k)

    def cdf(self, k):
        """Calculate cdf"""
        try:
            k = int(k)
            assert 0 <= k <= self.n
        except (TypeError, AssertionError):
            return 0
        return sum(self.pmf(i) for i in range(k+1))
