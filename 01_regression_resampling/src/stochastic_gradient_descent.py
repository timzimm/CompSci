import numpy as np
from collections import deque  # fixed size FIFO queue
import types


class StochasticGradientDescent:
    def __init__(
        self,
        max_epochs=1000,
        batches=None,
        shuffle=True,
        seed=42,
        learning_rate=lambda t, grad: 1e-3,
        momentum=0,
        n_iter_no_change=5,
        tol=0.001,
    ):
        self.max_epochs = max_epochs
        self.batches = batches
        self.shuffle = shuffle
        self.seed = seed
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.iteration = 0

    def minimize(self, loss_function, gradient_loss, X, y):
        # Toggle for reproducability
        if self.shuffle:
            np.random.seed(self.seed)

        if self.batches is None:
            # Assume vanilla stochastic gradient descent, i.e.
            # each minibatch has only one element
            self.batches = X.shape[0]

        # global minimization time, passed to the learning_rate functor
        self.iteration = 1
        epochs_wo_improvement = 0

        # fiducial initial loss
        best_loss = np.inf
        # initial guess for parameters ~ U(0,1)
        p = best_p = np.random.rand(X.shape[1])

        step = 0
        for epoch in range(self.max_epochs):
            # initial index array from which minibatches will be formed
            mini_batches = np.arange(X.shape[0])

            if self.shuffle:
                np.random.shuffle(mini_batches)

            if X.shape[0] % self.batches != 0:
                mini_batch_indices = np.array_split(mini_batches, self.batches)
            else:
                mini_batch_indices = mini_batches.reshape(self.batches, -1)

            # Learning rate is set by external, user-provided policy, dependent on
            # optimization iteration (for e.g. inverse scaling) and/or past gradient
            # (for e.g. RMSProp, ADAM, etc.).
            for mini_batch_idx in mini_batch_indices:
                # Advanced indexing implies a deep-copy. Thus, the order of
                # observations in X stays untouched.
                X_b = X[mini_batch_idx, :]
                y_b = y[mini_batch_idx]

                grad = gradient_loss(X_b, y_b, p)

                # Two types of of user policies are supported:
                # (i) Generators with internal state (e.g. sum of past
                # gradients)
                if isinstance(self.learning_rate, types.GeneratorType):
                    next(self.learning_rate)
                    learning_rate = self.learning_rate.send(grad)
                # (ii) lambdas acting as pure function, i.e. no internal state
                else:
                    learning_rate = self.learning_rate(self.iteration, grad)

                step *= self.momentum
                step += learning_rate * grad

                p = p - step

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
                if current_loss > best_loss - self.tol:
                    epochs_wo_improvement += 1
                if epochs_wo_improvement == self.n_iter_no_change:
                    return best_p
                else:
                    best_loss = current_loss
                    best_p = p
                    epochs_wo_improvement = 0

        return p
