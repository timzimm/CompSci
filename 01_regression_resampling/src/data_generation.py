import numpy as np


def franke_function(x, y):
    """
    Franke Function

    Parameters:
        x (double): x coordinate
        y (double): y coordinate

    Returns:
        f(x,y) (double): Value of Franke function at (x,y)
    """
    return (
        3 / 4 * np.exp(-1 / 4 * (9 * x - 2) ** 2 - 1 / 4 * (9 * y - 2) ** 2)
        + 3 / 4 * np.exp(-1 / 49 * (9 * x + 1) ** 2 - 1 / 10 * (9 * y + 1))
        + 1 / 2 * np.exp(-1 / 4 * (9 * x - 7) ** 2 - 1 / 4 * (9 * y - 3) ** 2)
        - 1 / 5 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    )


def random_x_y_franke(N, sigma=0.1, seed=42):
    """
    Data Generation Process

    Draws N random samples (x,y) ~ U(0,1)U(0,1), evaluates the Franke
    function at this sights and adds some normal noise, epsilon, to the resulting
    function values

    Parameters:
        N (int): Number of datapoints (x,y)
        sigma (double, optional): standard deviation of epsilon
        seed (int, optional): Seed used to initial the random number generator

    Returns:
        x (1d array): random x-coordinates
        y (1d array): random y-coordinates
        z (1d array): f(x,y) + epsilon
    """
    # Draw N points form
    x = np.random.default_rng(seed).random(N)
    y = np.random.default_rng(seed + 1).random(N)
    z = franke_function(x, y) + np.random.default_rng(seed + 2).normal(0, sigma, N)
    return x, y, z


def generate_design_matrix(x, y, order):
    """
    Construct design matrix 'X' for two dimensional polynmial of degree 'order':

        f(x,y) = x + y + x^2 + xy + y^2 + ... + x y^(degree-1) + y^degree

    Note that we *omit* the constant intercept column by convention.

    Parameters:
        x (N x 1 array): All x-coordinates at which observations took place
        y (N x 1 array): All y-coordinates at which observations took place
        order (int): Polynomial order, i.e. highest exponent used in f(x,y)

    Returns:
        X (N x order array): Design matrix for polynmial of order p evaluated at
                             N sights (x,y)
    """
    assert x.shape[0] == y.shape[0]

    # Setup system matrix M by...
    # creating a list of multi-indices alpha with |alpha| = N
    alpha = []
    for n in range(order + 1):
        a = np.arange(n + 1)
        alpha += zip(a, n - a)

    # taking each data point to the power of the respective multindex andforming its product
    # M is now the matrix of all possible monomials (column) for each data point (row)
    xy = np.c_[x, y]
    M = np.power(xy[:, np.newaxis, :], alpha)
    return np.prod(M, axis=-1)[:, 1:]
