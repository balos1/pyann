"""
Various activation functions for neurons
"""

import numpy as np

def heaviside(c, x, derivative):
    """
    Implements the heaviside step function u_c(x).
    """
    if derivative:
        raise ValueError("Cannot differentiate Heavisde function")
    return 0.0 if x < c else 1.0


def logistic(c, x, derivative):
    """
    Implements the rectified linear function
    """
    fx = 1/(1+np.exp(-x))
    if derivative:
        return fx * (1 - fx)
    return fx


def relu(c, x, derivative):
    """
    Implements the rectified linear function
    """
    if derivative:
        return 1 if x > c else 0
    return 0.0 if x < c else x
