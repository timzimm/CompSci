from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import train_test_split
from collections import deque  # fixed size FIFO queue
import types
from scipy.special import expit

from stochastic_gradient_descent import StochasticGradientDescent


class RegressionBase(ABC):
    def __init__(self, fit_intercept=True):
        self.betas = None
        self.intercept = None
        self.fit_intercept = fit_intercept

    @abstractmethod
    def fit(self, X, y):
        """Fits the predictor given design matrix X and target vector y"""

    def predict(self, X):
        return X @ self.betas + self.intercept

    def _center_data(self, X, y):
        if not self.fit_intercept:
            return X, 0.0, y, 0.0

        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)

        return (X - X_mean), X_mean, y - y_mean, y_mean

    def _compute_intercept(self, X_mean, y_mean):
        if not self.fit_intercept:
            self.intercept = 0.0
        else:
            self.intercept = y_mean - X_mean.T @ self.betas


class ClassificationBase(ABC):
    def __init__(self):
        self.betas = None

    @abstractmethod
    def fit(self, X, y):
        """Fits the predictor given design matrix X and target vector y"""

    @abstractmethod
    def predict(self, X):
        """Predict dependent variable feature matrix X"""


class OrdinaryLeastSquare(RegressionBase):
    def __init__(self, fit_intercept=True, solver="pinv", **sgd_kwargs):
        super().__init__(fit_intercept)

        if not (solver == "pinv" or solver == "sgd"):
            raise ValueError("unsupported solver")

        self.solver = solver
        self.sgd = StochasticGradientDescent(**sgd_kwargs) if solver == "sgd" else None

    def _loss(self, X, y, p):
        y_pred = X @ p
        return np.dot(y - y_pred, y - y_pred)

    def _gradient_loss(self, X, y, p):
        y_pred = X @ p
        return 2 * X.T @ (y_pred - y)

    def fit(self, X, y):

        X, X_mean, y, y_mean = self._center_data(X, y)

        if self.solver == "pinv":
            self.betas = np.linalg.pinv(X.T @ X) @ X.T @ y
        if self.solver == "sgd":
            self.betas = self.sgd.minimize(self._loss, self._gradient_loss, X, y)

        self._compute_intercept(X_mean, y_mean)

        return self


class RidgeRegression(RegressionBase):
    def __init__(self, penalty, fit_intercept=True, solver="inv", **sgd_kwargs):
        super().__init__(fit_intercept)

        self.penalty = penalty
        if not (solver == "inv" or solver == "sgd"):
            raise ValueError("unsupported solver")

        self.solver = solver
        self.sgd = StochasticGradientDescent(**sgd_kwargs) if solver == "sgd" else None

    def _loss(self, X, y, p):
        y_pred = X @ p
        return np.dot(y - y_pred, y - y_pred) + self.penalty * np.dot(p, p)

    def _gradient_loss(self, X, y, p):
        y_pred = X @ p
        return 2 * (X.T @ (y_pred - y) + self.penalty * p)

    def fit(self, X, y):

        X, X_mean, y, y_mean = self._center_data(X, y)

        if self.solver == "inv":
            p = X.shape[-1]
            # Matrix inverse exists for all penalty > 0(symmetric + positive-definite)
            self.betas = np.linalg.inv(X.T @ X + self.penalty * np.eye(p, p)) @ X.T @ y
        if self.solver == "sgd":
            self.betas = self.sgd.minimize(self._loss, self._gradient_loss, X, y)

        self._compute_intercept(X_mean, y_mean)

        return self


class LogisticRegression(ClassificationBase):
    def __init__(self, penalty, **sgd_kwargs):
        self.penalty = penalty
        self.sgd = StochasticGradientDescent(**sgd_kwargs)

    def _loss(self, X, y, p):
        y_prop = expit(p[0] + X @ p[1:])
        div_0_guard = 10 * np.finfo(float).eps
        return -(
            np.dot(1 - y, np.log(div_0_guard + 1 - y_prop))
            + np.dot(y, np.log(div_0_guard + y_prop))
        )

    def _gradient_loss(self, X, y, p):
        y_y_prop = y - expit(p[0] + X @ p[1:])
        return -np.r_[np.sum(y_y_prop), X.T @ y_y_prop + 2 * self.penalty * p[1:]]

    def fit(self, X, y):
        beta0_betas = self.sgd.minimize(
            self._loss, self._gradient_loss, X, y, p=np.zeros(X.shape[1] + 1)
        )
        self.intercept = beta0_betas[0]
        self.betas = beta0_betas[1:]

        return self

    def predict(self, X):
        return np.heaviside(expit(X @ self.betas + self.intercept) - 0.5, 0).astype(int)
