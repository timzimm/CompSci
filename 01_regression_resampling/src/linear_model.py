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


class StochasticGradientDescent_Deprecated(RegressionBase):
    def __init__(
        self,
        loss="squared_error",
        max_epochs=1000,
        batches=None,
        shuffle=True,
        seed=42,
        learning_rate=lambda t, grad: 1e-3,
        momentum=0.9,
        early_stopping=False,
        validation_fraction=0.2,
        n_iter_no_change=5,
        tol=0.001,
        verbose=False,
        l2_penalty=0,
    ):
        super().__init__()
        self.loss = loss
        self.max_epochs = max_epochs
        self.batches = batches
        self.tol = tol
        self.shuffle = shuffle
        self.seed = seed
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose
        self.l2_penalty = l2_penalty

    def __loss(self, X, y):
        y_pred = self.predict(X)
        if self.loss == "squared_error":
            return np.dot(y - y_pred, y - y_pred)
        if self.loss == "ridge":
            return np.dot(y - y_pred, y - y_pred) + self.l2_penalty * np.dot(
                self.params, self.params
            )
        if self.loss == "cross_entropy":
            return -(
                np.dot(y, np.log(expit(y_pred)))
                + np.dot((1 - y), np.log(1 - expit(y_pred)))
            )

    def __gradient_loss(self, X, y):
        y_pred = self.predict(X)
        if self.loss == "squared_error":
            return 2 * X.T @ (y_pred - y)
        if self.loss == "ridge":
            return 2 * X.T @ (y_pred - y) + 2 * self.l2_penalty * self.params
        if self.loss == "cross_entropy":
            return -(X.T) @ (y - expit(y_pred)) + 2 * self.l2_penalty * self.params

    def fit(self, X, y):
        p = X.shape[-1]
        N = X.shape[0]

        if self.shuffle:
            np.random.seed(self.seed)

        # initial parameters ~ U(0,1)
        self.params = np.random.rand(p)

        if self.batches is None:
            # Assume vanilla stochastic gradient descent
            self.batches = N

        # Do not mess with the original data
        X_train = X_test = X
        y_train = y_test = y

        if self.early_stopping:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.validation_fraction, random_state=self.seed
            )

        # Initialize the score history such that the optimization proceeds for at
        # least n_iter_no_change epochs by assuming that "past" epochs have improved
        # the score by at least a decrement of tol each.
        score_history = deque(
            reversed(
                [
                    self.score(X_test, y_test) + i * self.tol
                    for i in range(self.n_iter_no_change)
                ]
            ),
            self.n_iter_no_change,
        )
        param_history = deque(
            self.n_iter_no_change * [self.params],
            self.n_iter_no_change,
        )

        self.iteration = 0
        step = 0
        learning_rate = 0
        for epoch in range(self.max_epochs):
            # Randomize training data on each epoch
            if self.shuffle:
                mini_batches = np.arange(X_train.shape[0])
                np.random.shuffle(mini_batches)
                mini_batch_indices = np.array_split(mini_batches, self.batches)

            for t, mini_batch_idx in enumerate(mini_batch_indices):
                x_b = X_train[mini_batch_idx, :]
                y_b = y_train[mini_batch_idx]

                # Learning rate set by external user-provided policy, dependent on
                # optimization iteration and gradient.
                grad = self.__gradient_loss(x_b, y_b)
                if isinstance(self.learning_rate, types.GeneratorType):
                    next(self.learning_rate)
                    learning_rate = self.learning_rate.send(grad)
                else:
                    learning_rate = self.learning_rate(self.iteration, grad)

                step *= self.momentum
                step += learning_rate * grad

                self.params = self.params - step

                self.iteration += 1

            param_history.appendleft(self.params)
            score_history.appendleft(self.score(X_test, y_test))
            delta_score = np.abs(score_history[0] - score_history[-1])
            if self.verbose:
                print(
                    f"Epoch {epoch}/{self.max_epochs}\t Score: {score_history[0]}\t dScore: {delta_score}"
                )

            # After each epoch we check for early convergence by computing the R2-score.
            # Notice if early_stopping=False, we use the entire training set to
            # compute the score (X_train = X_test = X)
            # Define convergence as improving R2 by less than tol over the last
            # n_iter_no_change epochs.
            # This includes the case of decreasing R2 (delta_score < 0)
            if delta_score < self.tol:
                # Pick best parameters within the last n_iter_no_change epochs
                self.param = param_history[np.argmax(score_history)]
                return self

        if self.verbose:
            print("Maximum number of epochs reached.")
