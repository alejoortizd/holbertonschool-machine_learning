#!/usr/bin/env python3
"""Determines if you should stop gradient descent early"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determines if you should stop gradient descent early"""
    check = opt_cost - cost > threshold
    if not check:
        count += 1
    else:
        count = 0
    return (count >= patience, count)
