#!/usr/bin/env python
import ray
import warnings
from sklearn_gp import GaussianProcessRegressor
from shifty_kernels import WhiteKernel, ConstantKernel, RBF
import pickle
import os

import numpy as np

from operator import itemgetter

from scipy.linalg import cholesky, cho_solve, solve_triangular

from sklearn.base import clone
from sklearn.utils import check_random_state

GPR_CHOLESKY_LOWER = True

ray.init()


@ray.remote
def predict_expert_(X, X_train_, y_train_, alpha_, L_, kernel_, return_std=False):

    K_trans = kernel_(X, X_train_)
    y_mean = K_trans @ alpha_

    V = solve_triangular(L_, K_trans.T, lower=GPR_CHOLESKY_LOWER, check_finite=False)

    if return_std:
        y_var = kernel_.diag(X)
        y_var -= np.einsum("ij,ji->i", V.T, V)

        y_var_negative = y_var < 0
        if np.any(y_var_negative):
            warnings.warn(
                "Predicted variances smaller than 0. " "Setting those variances to 0."
            )
            y_var[y_var_negative] = 0.0

        return y_mean, np.sqrt(y_var)
    else:
        return (y_mean,)


@ray.remote(num_returns=2)
def precompute_fixed_prediction_quantities_expert_(X_train_, y_train_, kernel_):

    K = kernel_(X_train_)
    try:
        Lk_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
    except np.linalg.LinAlgError as exc:
        exc.args = (
            f"The kernel, {kernel}, is not returning a positive "
            "definite matrix. Try gradually increasing the 'alpha' "
            "parameter of your GaussianProcessRegressor estimator.",
        ) + exc.args
        raise
    alphak_ = cho_solve(
        (Lk_, GPR_CHOLESKY_LOWER),
        y_train_,
        check_finite=False,
    )
    return alphak_, Lk_


@ray.remote
def log_marginal_likelihood_expert_(
    theta,
    X_train_,
    y_train_,
    kernel,
    alpha,
    eval_gradient=False,
    clone_kernel=True,
):

    kernel.theta = theta

    if eval_gradient:
        K, K_gradient = kernel(X_train_, eval_gradient=True)
    else:
        K = kernel(X_train_)

    K[np.diag_indices_from(K)] += alpha
    try:
        L = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
    except np.linalg.LinAlgError:
        return (-np.inf, np.zeros_like(theta)) if eval_gradient else -np.inf

    y_train = y_train_
    if y_train.ndim == 1:
        y_train = y_train[:, np.newaxis]

    alpha = cho_solve((L, GPR_CHOLESKY_LOWER), y_train, check_finite=False)

    log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
    log_likelihood_dims -= np.log(np.diag(L)).sum()
    log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
    log_likelihood = log_likelihood_dims.sum(axis=-1)

    if eval_gradient:
        inner_term = np.einsum("ik,jk->ijk", alpha, alpha)
        K_inv = cho_solve(
            (L, GPR_CHOLESKY_LOWER), np.eye(K.shape[0]), check_finite=False
        )
        inner_term -= K_inv[..., np.newaxis]
        log_likelihood_gradient_dims = 0.5 * np.einsum(
            "ijl,jik->kl", inner_term, K_gradient
        )
        log_likelihood_gradient = log_likelihood_gradient_dims.sum(axis=-1)

    if eval_gradient:
        return log_likelihood, log_likelihood_gradient
    else:
        return log_likelihood


class DistributedGaussianProcessRegressor(GaussianProcessRegressor):
    def __init__(
        self,
        M=1,
        kernel=None,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=False,
        random_state=None,
    ):
        super().__init__(
            kernel=kernel,
            alpha=0,  # Noise fitted (see addition of WhiteKernel below)
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=True,
            random_state=random_state,
        )

        self.M = min(M, os.cpu_count())

        if self.kernel is None:  # Use a RBF kernel as default
            self.kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
                1.0, length_scale_bounds="fixed"
            )

        self.kernel = self.kernel + WhiteKernel(noise_level=0.1)

    def fit(self, X, y):
        # Noise fitted by WhiteKernel
        self.alpha = 0
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

        partition = np.array_split(np.arange(X.shape[0]), self.M)
        self.X_train_ = []
        self.y_train_ = []
        for partition_k in partition:
            self.X_train_.append(ray.put(X[partition_k, :]))
            self.y_train_.append(ray.put(y[partition_k]))

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

        # Precompute predicition quantities only dependent on X_train
        self.precomp_alpha_L_k = [
            precompute_fixed_prediction_quantities_expert_.remote(
                self.X_train_[k], self.y_train_[k], self.kernel_
            )
            for k in range(self.M)
        ]

        # Infer noise from data but always predict latent function
        self.alpha = self.kernel_.k2.noise_level
        self.kernel_.k2.noise_level = 0

        return self

    def log_marginal_likelihood(self, theta, **kwargs):
        # Block until all experts reported their log marginal likelihood values
        llm_k_values = ray.get(
            [
                log_marginal_likelihood_expert_.remote(
                    theta,
                    self.X_train_[k],
                    self.y_train_[k],
                    self.kernel_,
                    self.alpha,
                    **kwargs,
                )
                for k in range(self.M)
            ]
        )

        if not kwargs["eval_gradient"]:
            return sum(llm_k_values)
        else:
            return tuple(map(sum, zip(*llm_k_values)))

    def predict(self, X, return_std=False, aggregation="rBCM"):
        if not hasattr(self, "X_train_"):
            # Unfitted predictions are not supported
            raise NotImplementedError(
                "DistributedGaussianProcessRegressor only "
                "predicts for trained kernels"
            )

        if self.M == 1:
            """
            No Aggregation. Just standard Gaussian Process Regression
            """
            return ray.get(
                predict_expert_.remote(
                    X,
                    ray.get(self.X_train_[0]),
                    ray.get(self.y_train_[0]),
                    *ray.get(self.precomp_alpha_L_k[0]),
                    self.kernel_,
                    return_std=return_std,
                )
            )

        sigma2_XX = self.kernel_.diag(X)
        mu = np.zeros(X.shape[0])
        sigma2 = np.zeros(X.shape[0])

        experts_running = [
            predict_expert_.remote(
                X,
                self.X_train_[k],
                self.y_train_[k],
                self.precomp_alpha_L_k[k][0],
                self.precomp_alpha_L_k[k][1],
                self.kernel_,
                return_std=True,
            )
            for k in range(self.M)
        ]

        if aggregation == "BCM":
            """
            Aggregation according to the Bayesian Committee Machine
            see Tresp, V. (2000). A Bayesian committee machine,
            Neural computation, 12(11), 2719-2741
            """
            while len(experts_running):
                experts_done, experts_running = ray.wait(experts_running)
                mu_k, sigma_k = ray.get(experts_done[0])
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
            while len(experts_running):
                experts_done, experts_running = ray.wait(experts_running)
                mu_k, sigma_k = ray.get(experts_done[0])
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

    def save(self, filename):
        self.filename = filename
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    def get_training_data_k_(self, k):
        return ray.get(self.X_train_[k]), ray.get(self.y_train_[k])

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

    M = 2
    kernel = ConstantKernel() * RBF(length_scale=0.1)

    print("start")
    dgp = DistributedGaussianProcessRegressor(kernel=kernel, M=M, **hyperparameters)
    dgp.fit(X_train, y_train)
    print(dgp.kernel_)

    gp = GaussianProcessRegressor(
        kernel=kernel + WhiteKernel(), alpha=0, **hyperparameters
    )
    gp.fit(X_train, y_train)
    gp.alpha = gp.kernel_.k2.noise_level
    gp.kernel_.k2.noise_level = 0
    print(gp.kernel_)

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
