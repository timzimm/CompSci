import numpy as np


def squared_error(data, prediction):
    data = data.flatten()
    prediction = prediction.flatten()
    return np.dot(data - prediction, data - prediction)


def mse(data, prediction):
    return 1 / data.shape[0] * squared_error(data, prediction)


def r_score(data, prediction):
    data = data.flatten()
    prediction = prediction.flatten()
    mean = np.mean(prediction)
    return 1 - data.shape[0] * mse(data, prediction) / np.dot(data - mean, data - mean)
