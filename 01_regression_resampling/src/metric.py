import numpy as np


def mse(data, model):
    data = data.flatten()
    model = model.flatten()
    return 1 / data.shape[0] * np.dot(data - model, data - model)


def r_score(data, model):
    data = data.flatten()
    model = model.flatten()
    mean = np.mean(model)
    return 1 - data.shape[0] * mse(data, model) / np.dot(data - mean, data - mean)
