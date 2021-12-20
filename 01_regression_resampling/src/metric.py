import numpy as np


def squared_error(data, prediction):
    return (data - prediction) ** 2


def mse(data, prediction):
    return np.mean(squared_error(data, prediction), axis=0)


def r_score(data, prediction):
    mean = np.mean(prediction)
    return 1 - data.shape[0] * mse(data, prediction) / np.dot(data - mean, data - mean)
