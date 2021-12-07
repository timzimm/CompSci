from abc import ABC, abstractmethod
import numpy as np


class Transformation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    @abstractmethod
    def fit_transform(self, X):
        pass


class Standardization(Transformation):
    def __init__(self):
        super().__init__()
        self.mu = 0
        self.sigma = 0

    def fit(self, X):
        if X.shape[1] > 1:
            self.mu = np.mean(X[:, 1:], axis=0)
            self.sigma = np.std(X[:, 1:], axis=0)
        return self

    def transform(self, X):
        if X.shape[1] > 1:
            return np.c_[(X[:, 0], (X[:, 1:] - self.mu) / self.sigma)]
        else:
            return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
