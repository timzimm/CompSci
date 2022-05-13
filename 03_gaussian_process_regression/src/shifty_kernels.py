import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform

from sklearn.utils.validation import _num_samples
from sklearn.exceptions import ConvergenceWarning

from sklearn.gaussian_process.kernels import (
    Hyperparameter,
    Kernel,
    NormalizedKernelMixin,
    StationaryKernelMixin,
    GenericKernelMixin,
    Sum,
    Product,
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
    """Constant kernel.

    Can be used as part of a product-kernel where it scales the magnitude of
    the other factor (kernel) or as part of a sum-kernel, where it modifies
    the mean of the Gaussian process.

    .. math::
        k(x_1, x_2) = constant\\_value \\;\\forall\\; x_1, x_2

    Adding a constant kernel is equivalent to adding a constant::

            kernel = RBF() + ConstantKernel(constant_value=2)

    is the same as::

            kernel = RBF() + 2


    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    constant_value : float, default=1.0
        The constant value which defines the covariance:
        k(x_1, x_2) = constant_value

    constant_value_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on `constant_value`.
        If set to "fixed", `constant_value` cannot be changed during
        hyperparameter tuning.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = RBF() + ConstantKernel(constant_value=2)
    >>> gpr = GaussianProcessRegressor(kernel=kernel, alpha=5,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.3696...
    >>> gpr.predict(X[:1,:], return_std=True)
    (array([606.1...]), array([0.24...]))
    """

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
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
            optional
            The gradient of the kernel k(X, X) with respect to the log of the
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
    """White kernel.

    The main use-case of this kernel is as part of a sum-kernel where it
    explains the noise of the signal as independently and identically
    normally-distributed. The parameter noise_level equals the variance of this
    noise.

    .. math::
        k(x_1, x_2) = noise\\_level \\text{ if } x_i == x_j \\text{ else } 0


    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    noise_level : float, default=1.0
        Parameter controlling the noise level (variance)

    noise_level_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'noise_level'.
        If set to "fixed", 'noise_level' cannot be changed during
        hyperparameter tuning.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = DotProduct() + WhiteKernel(noise_level=0.5)
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.3680...
    >>> gpr.predict(X[:2,:], return_std=True)
    (array([653.0..., 592.1... ]), array([316.6..., 316.6...]))
    """

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
            The gradient of the kernel k(X, X) with respect to the log of the
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
    """Radial-basis function kernel (aka squared-exponential kernel).

    The RBF kernel is a stationary kernel. It is also known as the
    "squared exponential" kernel. It is parameterized by a length scale
    parameter :math:`l>0`, which can either be a scalar (isotropic variant
    of the kernel) or a vector with the same number of dimensions as the inputs
    X (anisotropic variant of the kernel). The kernel is given by:

    .. math::
        k(x_i, x_j) = \\exp\\left(- \\frac{d(x_i, x_j)^2}{2l^2} \\right)

    where :math:`l` is the length scale of the kernel and
    :math:`d(\\cdot,\\cdot)` is the Euclidean distance.
    For advice on how to set the length scale parameter, see e.g. [1]_.

    This kernel is infinitely differentiable, which implies that GPs with this
    kernel as covariance function have mean square derivatives of all orders,
    and are thus very smooth.
    See [2]_, Chapter 4, Section 4.2, for further details of the RBF kernel.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    length_scale : float or ndarray of shape (n_features,), default=1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.

    References
    ----------
    .. [1] `David Duvenaud (2014). "The Kernel Cookbook:
        Advice on Covariance functions".
        <https://www.cs.toronto.edu/~duvenaud/cookbook/>`_

    .. [2] `Carl Edward Rasmussen, Christopher K. I. Williams (2006).
        "Gaussian Processes for Machine Learning". The MIT Press.
        <http://www.gaussianprocess.org/gpml/>`_

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import RBF
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = 1.0 * RBF(1.0)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9866...
    >>> gpc.predict_proba(X[:2,:])
    array([[0.8354..., 0.03228..., 0.1322...],
           [0.7906..., 0.0652..., 0.1441...]])
    """

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
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
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
    def __init__(self, kernel, offset):
        self.kernel = kernel
        self.offset = offset

    def get_params(self, deep=True):
        params = dict(kernel=self.kernel, offset=self.offset)
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
        return self.kernel.diag(X @ A.T)

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
