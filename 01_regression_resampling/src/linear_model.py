from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import train_test_split
from collections import deque  # fixed size FIFO queue
import types
from scipy.special import expit


class Predictor(ABC):
    def __init__(self):
        self.params = None

    @abstractmethod
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X @ self.params

    def score(self, X, y):
        model = self.predict(X)
        mean = np.mean(model)
        sq_error = np.dot(y - model, y - model)
        return 1 - sq_error / np.dot(y - mean, y - mean)


class OrdinaryLeastSquare(Predictor):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        self.params = np.linalg.pinv(X.T @ X) @ X.T @ y
        return self


class RidgeRegression(Predictor):
    def __init__(self, penalty):
        super().__init__()
        self.penalty = penalty

    def fit(self, X, y):
        p = X.shape[-1]
        # Matrix inverse exists for all  penalities > 0 (symmetric + positive-definit)
        self.params = np.linalg.inv(X.T @ X + self.penalty * np.eye(p, p)) @ X.T @ y
        return self


class StochasticGradientDescent(Predictor):
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
        l2_penalty=0
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
            return np.dot(y - y_pred, y - y_pred) + self.penalty * np.dot(
                self.params, self.params
            )
        if self.loss == 'cross_entropy':
            return -(np.dot(y,np.log(expit(y_pred)))+np.dot((1-y),np.log(1-expit(y_pred))))

    def __gradient_loss(self, X, y):
        y_pred = self.predict(X)
        if self.loss == "squared_error":
            return 2 * X.T @ (y_pred - y)
        if self.loss == "ridge":
            return 2 * X.T @ (y_pred - y) + 2 * self.penalty * self.params
        if self.loss == 'cross_entropy':
            return -(X.T)@(y-expit(y_pred)) + 2 * self.l2_penalty * self.params

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
