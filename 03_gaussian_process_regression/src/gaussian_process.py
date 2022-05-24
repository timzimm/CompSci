#!/usr/bin/env python
from sklearn_gp import GaussianProcessRegressor
from shifty_kernels import WhiteKernel, ConstantKernel, RBF
from multiprocessing import Pool
from functools import partial
import pickle
import os
from glob import glob

import numpy as np

from operator import itemgetter

from scipy.linalg import cholesky, cho_solve

from sklearn.base import clone
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn.utils import check_random_state

GPR_CHOLESKY_LOWER = True


class DistributedGaussianProcessRegressor(GaussianProcessRegressor):
    def __init__(
        self,
        M=1,
        kernel=None,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=False,
        copy_X_train=True,
        random_state=None,
    ):
        super().__init__(
            kernel=kernel,
            alpha=0,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=copy_X_train,
            random_state=random_state,
        )

        self.M = M
        self.master = True
        if self.kernel is None:  # Use a RBF kernel as default
            self.kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
                1.0, length_scale_bounds="fixed"
            )

        self.kernel = self.kernel + WhiteKernel(noise_level=0.1)
        self.alpha = 0

    def fit(self, X, y):
        """
        A modification of sklearn's fit()-function in GaussianProcessRegressor.
        Note that DistributedGaussianProcessRegressor finds ideal parameters by
        minimizing the product of log likelihoods:
                    log p(y|X, theta) = sum_k^M log p(y_k|X_k, theta)
        Crucially, all k=1..M experts share the *same* hyperparameters theta.

        A direct consequence is that we cannot push the fit implementation down
        to the expert level, by simply calling fit() for the k=1..M partitions of
        (X,y) as this would result in M independent hyperparameter sets.
        """
        self.kernel_ = clone(self.kernel)

        self._rng = check_random_state(self.random_state)

        if self.kernel_.requires_vector_input:
            dtype, ensure_2d = "numeric", True
        else:
            dtype, ensure_2d = None, False
        X, y = self._validate_data(
            X,
            y,
            multi_output=True,
            y_numeric=True,
            ensure_2d=ensure_2d,
            dtype=dtype,
        )

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            self._y_train_std = _handle_zeros_in_scale(np.std(y, axis=0), copy=False)

            # Remove mean and make unit variance
            y = (y - self._y_train_mean) / self._y_train_std

        else:
            shape_y_stats = (y.shape[1],) if y.ndim == 2 else 1
            self._y_train_mean = np.zeros(shape=shape_y_stats)
            self._y_train_std = np.ones(shape=shape_y_stats)

        if np.iterable(self.alpha) and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError(
                    "alpha must be a scalar or an array with same number of "
                    f"entries as y. ({self.alpha.shape[0]} != {y.shape[0]})"
                )

        self.M = min(X.shape[0], self.M, os.cpu_count())
        self.partition = np.array_split(np.arange(X.shape[0]), self.M)

        # Shared-memory for trainings data
        self.X_train_shm_ = np.memmap("X_train", X.dtype, "w+", shape=X.shape)
        self.y_train_shm_ = np.memmap("y_train", y.dtype, "w+", shape=y.shape)
        self.X_train_shm_[:] = X
        self.y_train_shm_[:] = y

        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)

            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True, clone_kernel=False
                    )
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta, clone_kernel=False)

            # First optimize starting from theta specified in kernel
            optima = [
                (
                    self._constrained_optimization(
                        obj_func, self.kernel_.theta, self.kernel_.bounds
                    )
                )
            ]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0)"
                        "requires that all bounds are finite."
                    )
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial, bounds)
                    )
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.kernel_._check_bounds_params()

            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = self.log_marginal_likelihood(
                self.kernel_.theta, clone_kernel=False
            )

        # Precompute predicition quantities only dependent on X_train
        if self.M == 1:
            # Spare the pool spin-up overhead if a vanilla GP is requested
            self.precompute_fixed_prediction_quantities_k_(0)
        else:
            with Pool() as pool:
                pool.map(
                    self.precompute_fixed_prediction_quantities_k_, np.arange(self.M)
                )

        # Infer noise from data but always predict latent function
        self.alpha = self.kernel_.k2.noise_level
        self.kernel_.k2.noise_level = 0

        return self

    def precompute_fixed_prediction_quantities_k_(self, k):
        self.master = False

        self.X_train_, self.y_train_ = self.get_training_data_k_(k)

        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            Lk_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                f"The kernel, {self.kernel_}, is not returning a positive "
                "definite matrix. Try gradually increasing the 'alpha' "
                "parameter of your GaussianProcessRegressor estimator.",
            ) + exc.args
            raise
        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        alphak_ = cho_solve(
            (Lk_, GPR_CHOLESKY_LOWER),
            self.y_train_,
            check_finite=False,
        )
        np.save(f"./L_k_{k}.npy", Lk_, allow_pickle=False)
        np.save(f"./alpha_k_{k}.npy", alphak_, allow_pickle=False)

    def log_marginal_likelihood(
        self, theta=None, eval_gradient=False, clone_kernel=True
    ):
        cost_per_k = partial(
            self.log_marginal_likelihood_k_,
            theta=theta,
            eval_gradient=eval_gradient,
            clone_kernel=clone_kernel,
        )
        if self.M == 1:
            # Spare the pool spin-up overhead if a vanilla GP is requested
            llm_k_values = cost_per_k(0)
            self.master = True
            return llm_k_values

        else:
            # This may or may not cause a lot of overhead as we spin up a new pool per
            # call to log_marginal_likelihood which happens often during optimization.
            # An optimized implementation should call Pool() outside of the
            # optimization loop and synchronize processes per function call. That
            # said, although not perfect, the overhead should amortize as the
            # problem size grows.
            with Pool() as pool:
                llm_k_values = pool.map(cost_per_k, np.arange(self.M))

        if not eval_gradient:
            return sum(llm_k_values)
        else:
            return tuple(map(sum, zip(*llm_k_values)))

    def log_marginal_likelihood_k_(self, k, **kwargs):
        self.master = False

        self.X_train_, self.y_train_ = self.get_training_data_k_(k)

        return super().log_marginal_likelihood(**kwargs)

    def predict(self, X, return_std=False, return_cov=False, aggregation="rBCM"):
        if not hasattr(self, "X_train_shm_"):
            raise NotImplementedError(
                "DistributedGaussianProcessRegressor only "
                "predicts for trained kernels"
            )
        if return_cov:
            raise NotImplementedError(
                "Covariances not handled by DistributedGaussianProcessRegressor"
            )

        if self.M == 1:
            """
            No Aggregation. Just standard Gaussian Process Regression
            """
            return self.predict_k(0, X, return_std=return_std)

        sigma2_XX = self.kernel_.diag(X)
        mu = np.zeros(X.shape[0])
        sigma2 = np.zeros(X.shape[0])

        if aggregation == "BCM":
            """
            Aggregation according to the Bayesian Committee Machine
            see Tresp, V. (2000). A Bayesian committee machine,
            Neural computation, 12(11), 2719-2741
            """
            for k in range(self.M):
                mu_k, sigma_k = self.predict_k(k, X, return_std=True)

                sigma2 += sigma_k ** (-2)
                mu += sigma_k ** (-2) * mu_k

            sigma2 += (1 - self.M) * sigma2_XX ** (-1)
        elif aggregation == "rBCM":
            """
            Aggregation according to the *robust* Bayesian Committee Machine
            see Deisenroth, M. P., & Ng, J. W. (2015). Distributed gaussian processes,
            arXiv:1502.02843.
            """
            beta = 0
            for k in range(self.M):
                mu_k, sigma_k = self.predict_k(k, X, return_std=True)
                beta_k = 0.5 * (np.log(sigma2_XX) - np.log(sigma_k**2))

                beta_sigma2_k = beta_k * sigma_k ** (-2)
                sigma2 += beta_sigma2_k
                mu += beta_sigma2_k * mu_k
                beta += beta_k

            sigma2 += (1 - beta) * sigma2_XX ** (-1)

        sigma2 = 1.0 / sigma2
        mu *= sigma2
        if return_std:
            return mu, np.sqrt(sigma2)
        else:
            return mu

    def predict_k(self, k, X, **kwargs):

        # Load precomputed data of expert k and populate class members that the
        # serial GP base class uses for the prediction, i.e. the class members
        # usually used to hold cached data. This makes a call to the
        # base class implementation possible w/o additional tweaks.
        self.L_ = np.load(f"./L_k_{k}.npy", allow_pickle=False)
        self.alpha_ = np.load(f"./alpha_k_{k}.npy", allow_pickle=False)
        self.X_train_, self.y_train_ = self.get_training_data_k_(k)

        return super().predict(X, **kwargs)

    def get_training_data_k_(self, k):
        # Advanced indexing implies a copy! This will degrade performance.
        return (
            self.X_train_shm_[self.partition[k], :],
            self.y_train_shm_[self.partition[k]],
        )

    def save(self, filename):
        self.filename = filename
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename=None):
        filename = cls.filename if filename is None else filename

        with open(filename, "rb") as file:
            gp = pickle.load(file)
            if type(gp) is not DistributedGaussianProcessRegressor:
                raise TypeError(
                    "Deserialized object is not a"
                    "DistributedGaussianProcessRegressor!"
                )

        return gp

    def __del__(self):

        # Only the parent process is allowed to delete temporary (cached) files.
        if self.master:
            for f in glob("*.npy"):
                os.remove(f)
            os.remove("X_train")
            os.remove("y_train")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    def generate_data(x, sigma=0.25, seed=42):
        N = x.shape[0]
        return (
            5 * x**2 * np.sin(12 * x)
            + (x**3 - 0.5) * np.sin(3 * x - 0.5)
            + 4 * np.cos(2 * x)
            + np.random.default_rng(seed=seed).normal(0, scale=sigma, size=N)
        )

    N = 1000
    Ntest = 100
    X_train = np.random.rand(N)
    X_test = np.linspace(-1.5, 2.5, Ntest)

    y_train = generate_data(X_train)
    y_train_mean = np.mean(y_train)
    y_train = y_train - y_train_mean
    y_test = generate_data(X_test) - y_train_mean

    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    hyperparameters = {"n_restarts_optimizer": 0}

    M = 4
    kernel = ConstantKernel() * RBF(length_scale=0.1)

    gp = GaussianProcessRegressor(
        kernel=kernel + WhiteKernel(), alpha=0, **hyperparameters
    )
    gp.fit(X_train, y_train)
    gp.alpha = gp.kernel_.k2.noise_level
    gp.kernel_.k2.noise_level = 0
    print(gp.kernel_)

    dgp = DistributedGaussianProcessRegressor(kernel=kernel, M=M, **hyperparameters)
    dgp.fit(X_train, y_train)
    print(dgp.kernel_)

    mu, sigma = gp.predict(X_test, return_std=True)
    mu_dgp, sigma_dgp = dgp.predict(X_test, return_std=True, aggregation="rBCM")

    fig, ax = plt.subplots()

    sns.set_palette("Blues", M)
    sns.set_palette("Blues", M)
    for k in range(M):
        X_train_k, y_train_k = dgp.get_training_data_k_(k)
        X_train_k = X_train_k.squeeze()
        ax.scatter(X_train_k, y_train_k, alpha=0.6, s=8)
    ax.plot(X_test, mu, color="k")
    ax.plot(X_test, mu + sigma, color="k", ls="dashed")
    ax.plot(X_test, mu - sigma, color="k", ls="dashed")
    ax.plot(X_test, mu_dgp, color="red")
    ax.fill_between(
        X_test.squeeze(), mu_dgp - sigma_dgp, mu_dgp + sigma_dgp, alpha=0.3, color="red"
    )
    ax.set_xlim([X_test.min(), X_test.max()])
    plt.show()
    del dgp
