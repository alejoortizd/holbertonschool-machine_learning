#!/usr/bin/env python3
"""Calculates the precision for each
class in a confusion matrix"""
import numpy as np


def precision(confusion):
    """Calculates the precision for each
    class in a confusion matrix"""
    TP = np.diag(confusion)
    FP = confusion.sum(axis=0) - np.diag(confusion)
    PPV = TP/(TP+FP)
    return PPV
