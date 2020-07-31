#!/usr/bin/env python3
"""Create a class Exponential that represents a exponential distribution"""


def erf(x, n=4):
    """Compute the Maclaurin approximation of erf"""
    π = 3.1415926536
    return 2/π**0.5*sum([x, -1/3*x**3, 1/10*x**5, -1/42*x**7, 1/216*x**9])


class Normal:
    """Represent the normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize normal distribution"""
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None and self.stddev <= 0:
            raise ValueError('stddev must be a positive value')
        if data is not None:
            if type(data) != list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = sum(data)/len(data)
            self.stddev = (sum((self.mean - x)**2 for x in data)/len(data))**.5

    def z_score(self, x):
        """Calculate the z-score"""
        return (x - self.mean)/self.stddev

    def x_value(self, z):
        """Calculate the value"""
        return z*self.stddev + self.mean

    def pdf(self, x):
        """Calculate the pdf"""
        π = 3.1415926536
        e = 2.7182818285
        return 1/(self.stddev * (2*π)**0.5) * e**(-1/2 * self.z_score(x)**2)

    def cdf(self, x):
        """Calculate the cdf"""
        return 0.5*(1 + erf(self.z_score(x)/2**0.5))
