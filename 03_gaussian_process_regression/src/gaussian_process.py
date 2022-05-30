#!/usr/bin/env python
from sklearn_gp import GaussianProcessRegressor
from shifty_kernels import WhiteKernel
import pickle
import numpy as np


class NoiseFittedGP(GaussianProcessRegressor):
    """Trivial GaussianProcessRegressor derivative that infers the noise of the data
    but predicts the latent function :math:`f` instead of the noisy :math:`y`.
    It is assumed that the kernel passed in has no
    WhiteKernel contribution for modelling iid gaussian noise.
    This will be added by NoiseFittedGP.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel = self.kernel + WhiteKernel(noise_level=0.1)
        self.kernel_ = self.kernel
        self.alpha = 0

    def fit(self, X=None, y=None):
        """Consult sklearn documentation of GaussianProcessRegressor"""
        if X is not None and y is not None:
            self.set_training_data(X, y)
        elif (X is None or y is None) and not hasattr(self, "X_train_"):
            raise ValueError("training data unspecified")
        elif X is None ^ y is None:
            raise NotImplementedError("Either provide both X and y or nothing")

        copy_X_train = self.copy_X_train
        self.copy_X_train = False
        super().fit(self.X_train_, self.y_train_)
        self.copy_X_train = copy_X_train

        self.alpha = self.kernel_.k2.noise_level
        self.kernel_.k2.noise_level = 0
        return self

    def set_training_data(self, X, y):
        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

    def save(self, filename):
        """Serialize model and store it at filename.
        Useful for storing trained model.
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        """Load serialized model and return a new NoiseFittedGP instance"""

        with open(filename, "rb") as file:
            gp = pickle.load(file)
            if type(gp) is not NoiseFittedGP:
                raise TypeError("Deserialized object is not a NoiseFittedGP!")

        return gp
