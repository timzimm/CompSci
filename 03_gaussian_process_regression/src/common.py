import numpy as np
from numpy.random import default_rng


def noisy_friedman(x, sigma=0.1, seed=42):
    assert x.shape[1] == 5
    N = x.shape[0]
    f = (
        10 * np.sin(np.pi * x[:, 0] * x[:, 1])
        + 20 * (x[:, 2] - 0.5) ** 2
        + 10 * x[:, 3]
        + 5 * x[:, 4]
    )

    epsilon = default_rng(seed).normal(0, sigma, N)
    return f + epsilon

# Generates trivial projection matrix in x_{direction} 
def P(dim, direction):
    P_x = np.zeros((dim,dim))
    P_x[direction,direction] = 1
    return P_x