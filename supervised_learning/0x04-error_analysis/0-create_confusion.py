#!/usr/bin/env python3
"""Creates a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix"""
    confMatrix = np.matmul(labels.T, logits)
    return confMatrix
