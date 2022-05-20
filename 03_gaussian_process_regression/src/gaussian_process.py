#!/usr/bin/env python
from sklearn.gaussian_process import GaussianProcessRegressor
from shifty_kernels import WhiteKernel
import pickle


class NoiseFittedGP(GaussianProcessRegressor):
    """Trivial GaussianProcessRegressor derivative that infers the noise of the data
    but predicts the latent function :math:`f` instead of the noisy :math:`y`.
    It is assumed that the kernel passed in has no
    WhiteKernel contribution for modelling iid gaussian noise.
    This will be added by NoiseFittedGP.
    """

    def __init__(self, M=8 * args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel = self.kernel + WhiteKernel(noise_level=0.1)
        self.alpha = 0

    def fit(self, X, y):
        """Consult sklearn documentation of GaussianProcessRegressor
        We modify the behavior by extracting the noise level from the fitted
        kernel, set this as iid noise, and hence not inflating the posterior
        variance, and disable the WhiteKernel
        """
        super().fit(X, y)

        self.alpha = self.kernel_.k2.noise_level
        self.kernel_.k2.noise_level = 0
        return self

    def log_marginal_likelihood_k(
        self, k, theta=None, eval_gradient=False, clone_kernel=True
    ):
        pass

    def save(self, filename):
        """Serialzie model and store it at filename.
        Useful for storing trained model.
        """
        self.filename = filename
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename=None):
        """Load serialized model and return a new NoiseFittedGP instance"""
        filename = self.filename if filename is None else filename

        with open(filename, "rb") as file:
            gp = pickle.load(file)
            if type(gp) is not NoiseFittedGP:
                raise TypeError("Deserialized object is not a NoiseFittedGP!")

        return gp


if __name__ == "__main__":
    pass
