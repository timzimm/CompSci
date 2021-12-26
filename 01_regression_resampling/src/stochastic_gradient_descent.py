import numpy as np
from collections import deque  # fixed size FIFO queue
import types
from abc import ABC, abstractmethod


class Scheduler:
    def __init__(self, log):
        self.log = log
        self.learning_rates = []

    def __log(self, learning_rate):
        if self.log:
            self.learning_rates.append(learning_rate)
        return learning_rate


class Constant(Scheduler):
    def __init__(self, log=False, eta0=0.01):
        super().__init__(log)
        self.eta0 = eta0

    def __call__(self, t, grad):
        return self._Scheduler__log(self.eta0)


class InverseScaling(Scheduler):
    def __init__(self, log=False, eta0=0.01, power_t=0.125):
        super().__init__(log)
        self.eta0 = eta0
        self.power_t = power_t

    def __call__(self, t, grad):
        return self._Scheduler__log(self.eta0 / t ** self.power_t)


class AdaGrad(Scheduler):
    def __init__(self, log=False, eta0=0.01, epsilon=1e-8):
        super().__init__(log)
        self.eta0 = eta0
        self.epsilon = epsilon
        self.g2_t = 0

    def __call__(self, t, grad):
        self.g2_t += grad ** 2
        return self._Scheduler__log(self.eta0 / np.sqrt(self.epsilon + self.g2_t))


class RMSProp(Scheduler):
    def __init__(self, log=False, eta0=0.01, beta=0.9, epsilon=1e-8):
        super().__init__(log)
        self.eta0 = eta0
        self.beta = beta
        self.epsilon = epsilon
        self.g2_t_avg = 0

    def __call__(self, t, grad):
        self.g2_t_avg = self.beta * self.g2_t_avg + (1 - self.beta) * grad ** 2
        return self._Scheduler__log(self.eta0 / np.sqrt(self.epsilon + self.g2_t_avg))


class StochasticGradientDescent:
    def __init__(
        self,
        max_epochs=1000,
        batches=None,
        shuffle=True,
        seed=42,
        learning_rate=Constant(),
        n_iter_no_change=5,
        tol=0.001,
    ):
        self.max_epochs = max_epochs
        self.batches = batches
        self.shuffle = shuffle
        self.seed = seed
        self.learning_rate = learning_rate
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.iteration = 0

    def minimize(self, loss_function, gradient_loss, X, y):

        # Toggle for reproducability
        if self.shuffle:
            np.random.seed(self.seed)

        N = X.shape[0]
        if self.batches is None:
            # Assume vanilla stochastic gradient descent, i.e.
            # each minibatch has only one element
            self.batches = N

        # global minimization time, passed to the learning_rate functor
        self.iteration = 1
        epochs_wo_improvement = 0

        # fiducial initial loss
        best_loss = np.inf

        # Same initial guess as in SGDRegressor
        p = best_p = np.zeros(X.shape[1])

        minibatch_boundaries = np.arange(0, N, N // self.batches)
        if N % self.batches == 0:
            minibatch_boundaries = np.hstack((minibatch_boundaries, N))
        else:
            minibatch_boundaries[-1] = N

        # Merge X and y along the last dimension for combined shuffle.
        # Note we could also use an 1D-array holding indices, say idx, and shuffle
        # only idx. Advanced indexing, i.e. X[idx,:] and y[idx], would then
        # yield the sought after, randomized mini batches. Unfortunately,
        # advanced indexing implies a copy (and does not return a view), which
        # makes a call to minimize roughly 18% slower if self.batches = N
        # (we profiled it...a lot...)
        Xy = np.c_[X, y]
        step = 0
        for epoch in range(self.max_epochs):

            if self.shuffle:
                # For reasons not enirely clear, using permutation instead of
                # shuffle is 33% faster (compared to using shuffle) ¯\_(ツ)_/¯
                Xy = np.random.permutation(Xy)

            for i in range(minibatch_boundaries.shape[0] - 1):
                start = minibatch_boundaries[i]
                end = minibatch_boundaries[i + 1]
                X_b = Xy[start:end, :-1]
                y_b = Xy[start:end, -1]

                mean_minibatch_grad = 1.0 / X_b.shape[0] * gradient_loss(X_b, y_b, p)

                # Learning rate is set by external, user-provided policy, dependent on
                # optimization iteration (for e.g. inverse scaling) and/or past
                # gradient (for e.g. RMSProp, ADAM, etc.).
                p -= (
                    self.learning_rate(self.iteration, mean_minibatch_grad)
                    * mean_minibatch_grad
                )

                self.iteration += 1

            # Early converence test if user provides an absolut tolerance.
            # If not (i.e. self.tol=None), descent proceeds until max_epochs is reached
            if self.tol:
                # Check convergence after each epoch. In accordance with sklearn,
                # we define convergence being achieved, if
                #               loss_function(X, y) > best_loss - tol
                # is satisfied for n_iter_no_change epochs. Note that:
                # (i) the *entire* trainings data is used to compute the loss
                current_loss = loss_function(X, y, p)
                if current_loss > best_loss - self.tol * N:
                    epochs_wo_improvement += 1
                if epochs_wo_improvement == self.n_iter_no_change:
                    return best_p
                else:
                    best_loss = current_loss
                    best_p = p
                    epochs_wo_improvement = 0

        return p
