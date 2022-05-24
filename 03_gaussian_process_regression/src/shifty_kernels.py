# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD 3 clause
# Modified: Tim Zimmermann <timzi@uio.no>
# Modication Reason:
# Implementation of a SymmetricKernel requires taking derivatives of
# K(-X,X). This violates scikit-learns API contract.

import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform

from sklearn.utils.validation import _num_samples

from sklearn.gaussian_process.kernels import (
    Hyperparameter,
    Kernel,
    NormalizedKernelMixin,
    StationaryKernelMixin,
    GenericKernelMixin,
)


def _check_length_scale(X, length_scale):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError(
            "Anisotropic kernel must have the same number of "
            "dimensions as data (%d!=%d)" % (length_scale.shape[0], X.shape[1])
        )
    return length_scale


class ConstantKernel(StationaryKernelMixin, GenericKernelMixin, Kernel):
    def __init__(self, constant_value=1.0, constant_value_bounds=(1e-5, 1e5)):
        self.constant_value = constant_value
        self.constant_value_bounds = constant_value_bounds

    @property
    def hyperparameter_constant_value(self):
        return Hyperparameter("constant_value", "numeric", self.constant_value_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_X, n_features) or list of object, \
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
            optional
            The gradient of the kernel k(X, Y) with respect to the log of the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        Y = X if Y is None else Y
        if eval_gradient and Y.shape != X.shape:
            raise ValueError("Gradient can only be evaluated when Y.shape=X.shape.")

        K = np.full(
            (_num_samples(X), _num_samples(Y)),
            self.constant_value,
            dtype=np.array(self.constant_value).dtype,
        )
        if eval_gradient:
            if not self.hyperparameter_constant_value.fixed:
                return (
                    K,
                    np.full(
                        (_num_samples(X), _num_samples(Y), 1),
                        self.constant_value,
                        dtype=np.array(self.constant_value).dtype,
                    ),
                )
            else:
                return K, np.empty((_num_samples(X), _num_samples(Y), 0))
        else:
            return K

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return np.full(
            _num_samples(X),
            self.constant_value,
            dtype=np.array(self.constant_value).dtype,
        )

    def __repr__(self):
        return "{0:.3g}**2".format(np.sqrt(self.constant_value))


class WhiteKernel(StationaryKernelMixin, GenericKernelMixin, Kernel):
    def __init__(self, noise_level=1.0, noise_level_bounds=(1e-5, 1e5)):
        self.noise_level = noise_level
        self.noise_level_bounds = noise_level_bounds

    @property
    def hyperparameter_noise_level(self):
        return Hyperparameter("noise_level", "numeric", self.noise_level_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_X, n_features) or list of object,\
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),\
            optional
            The gradient of the kernel with respect to the log of the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if Y is not None and eval_gradient:
            if Y.shape != X.shape:
                raise ValueError(
                    "Gradient can only be evaluated when Y=Y.shape=X.shape."
                )

        if Y is None:
            K = self.noise_level * np.eye(_num_samples(X))
        else:
            K = np.zeros((_num_samples(X), _num_samples(Y)))

        if eval_gradient:
            if not self.hyperparameter_noise_level.fixed and Y is None:
                return (
                    K,
                    self.noise_level * np.eye(_num_samples(X))[:, :, np.newaxis],
                )
            else:
                return K, np.empty((_num_samples(X), _num_samples(Y), 0))
        else:
            return K

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return np.full(
            _num_samples(X), self.noise_level, dtype=np.array(self.noise_level).dtype
        )

    def __repr__(self):
        return "{0}(noise_level={1:.3g})".format(
            self.__class__.__name__, self.noise_level
        )


class RBF(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale),
            )
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)

        if Y is None:
            dists = pdist(X / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient and Y.shape != X.shape:
                raise ValueError("Gradient can only be evaluated when Y.shape=X.shape.")
            dists = cdist(X / length_scale, Y / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)

        if eval_gradient:
            Y = X if Y is None else Y
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], Y.shape[0], 0))
            else:
                K_gradient = (X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2 / (
                    length_scale**2
                )
                if not self.anisotropic or length_scale.shape[0] == 1:
                    K_gradient = np.sum(K_gradient, axis=-1, keepdims=True)
                K_gradient *= K[..., np.newaxis]
            return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
            )
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0]
            )


class SymmetricKernel1D(Kernel):
    """The `SymmetricKernel1D` kernel takes one kernel :math:`k_1` and an
    offset `o` and symmetrizes `k_1` around `o` via:
    .. math::
        k_{sym}(X, Y) = 0.5 * (k_1(-(X-o), Y-o) + k_1(X-o, Y-o))
    functions drawn from a GP with `k_{sym}` are guaranteed to satisfy `f(x) =
    f(-x)`. Not that SymmetricKernel1D is meant to be used for one dimensional
    input spaces. Higher dimensional input spaces can be projected to lower
    dimensional sub-spaces via `ProjectionKernel` and a composition of
    `ProjectionKernel` and `SymmetricKernel1D` is therfore a canoncial use case.
    Currently, `o` is not considered to be a tunable hyperparameter. Promoting
    it to such would require to implement gradients with respect to the inputs
    of the correlation function.
    Parameters
    ----------
    k1 : Kernel
        The base-kernel to symmetrize
    k2 : Kernel
        Center of symmetry
    """

    def __init__(self, kernel, offset):
        self.kernel = kernel
        self.offset = offset

    @property
    def hyperparameters(self):
        r = []
        for hyperparameter in self.kernel.hyperparameters:
            r.append(
                Hyperparameter(
                    "kernel__" + hyperparameter.name,
                    hyperparameter.value_type,
                    hyperparameter.bounds,
                    hyperparameter.n_elements,
                )
            )
        return r

    @property
    def anisotropic(self):
        return False

    def is_stationary(self):
        return False

    @property
    def theta(self):
        return self.kernel.theta

    @theta.setter
    def theta(self, theta):
        self.kernel.theta = theta

    @property
    def bounds(self):
        return self.kernel.bounds

    def __eq__(self, b):
        if type(self) != type(b):
            return False
        return self.kernel == b.kernel

    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient:
            if Y is not None:
                raise ValueError("Gradient is only evaluated when Y = X")
            K1, K1_gradient = self.kernel(
                -(X - self.offset), X - self.offset, eval_gradient=True
            )
            K2, K2_gradient = self.kernel(
                X - self.offset, X - self.offset, eval_gradient=True
            )
            return 0.5 * (K1 + K2), 0.5 * (K1_gradient + K2_gradient)
        else:
            Y = X if Y is None else Y
            K1 = self.kernel(-(X - self.offset), Y - self.offset)
            K2 = self.kernel(X - self.offset, Y - self.offset)
            return 0.5 * (K1 + K2)

    def diag(self, X):
        return 0.5 * (
            np.diag(self.kernel(-(X - self.offset), (X - self.offset)))
            + self.kernel.diag(X - self.offset)
        )

    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__, self.kernel.__repr__())


class ProjectionKernel(Kernel):
    """The `ProjectionKernel` takes one kernel :math:`k` and a
    projection matrix :math:`A` of rank :math:`M` and projects a higher dimensional
    input space of dimension :math:`N` to the subspace encoded in :math:`A`:
    .. math::
        k_{P}(X, Y) = k(AX, AY)
    Parameters
    ----------
    k : Kernel
        The kernel applied in the lower dimensional subspace
    A : Projection Matrix
        Matrix with property :math:`AA = A`
    """

    def __init__(self, kernel, A, tag=None):
        if not np.allclose(A @ A, A):
            raise ValueError("A is not a projection matrix")
        self.A = A
        self.tag = tag
        self.kernel = kernel

    def get_params(self, deep=True):
        params = dict(kernel=self.kernel, A=self.A, tag=self.tag)
        if deep:
            deep_items = self.kernel.get_params().items()
            params.update(("kernel__" + k, val) for k, val in deep_items)
        return params

    @property
    def hyperparameters(self):
        r = []
        for hyperparameter in self.kernel.hyperparameters:
            r.append(
                Hyperparameter(
                    "kernel__" + hyperparameter.name,
                    hyperparameter.value_type,
                    hyperparameter.bounds,
                    hyperparameter.n_elements,
                )
            )
        return r

    @property
    def anisotropic(self):
        return self.kernel.anisotropic

    def is_stationary(self):
        return self.kernel.is_stationary()

    @property
    def theta(self):
        return self.kernel.theta

    @theta.setter
    def theta(self, theta):
        self.kernel.theta = theta

    @property
    def bounds(self):
        return self.kernel.bounds

    def __eq__(self, b):
        if type(self) != type(b):
            return False
        return self.kernel == b.kernel and self.A == b.A and self.tag == B.tag

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            return self.kernel(X @ self.A.T, None, eval_gradient=eval_gradient)
        else:
            return self.kernel(X @ self.A.T, Y @ self.A.T, eval_gradient=eval_gradient)

    def diag(self, X):
        return self.kernel.diag(X @ self.A.T)

    def __repr__(self):
        return "{0}({1}, {2})".format(
            self.__class__.__name__, self.tag, self.kernel.__repr__()
        )


if __name__ == "__main__":

    from sklearn.gaussian_process import GaussianProcessRegressor
    import matplotlib.pyplot as plt
    from scipy.stats.qmc import LatinHypercube

    test_symmetric = True
    if test_symmetric:
        X = np.linspace(-1, 1, 200).reshape(-1, 1)
        sqexp = RBF(length_scale=0.1)
        sqexp_sym = SymmetricKernel1D(sqexp, offset=0.4)
        gp = GaussianProcessRegressor(kernel=sqexp_sym)
        print(gp.kernel)
        for i in range(5):
            Y = gp.sample_y(X, random_state=i)
            plt.plot(X.flatten(), Y.flatten())

    test_composition = False
    if test_composition:
        dim = 5
        sqexp = RBF(length_scale=0.1)
        P0 = np.zeros((dim, dim))
        P0[0, 0] = 1
        kernel_x0 = ProjectionKernel(sqexp, P0, tag="x0")
        sym_x0 = SymmetricKernel1D(kernel_x0)

        dim = 5
        N = 100
        X = LatinHypercube(dim).random(N) - 0.5
        gp = GaussianProcessRegressor(kernel=sym_x0)
        for i in range(1):
            Y = gp.sample_y(X, random_state=i)
            plt.scatter(X[:, 0], Y.flatten())

    plt.show()
