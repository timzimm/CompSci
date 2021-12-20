import numpy as np
from metric import mse, squared_error
from sklearn.model_selection import train_test_split


def prediction_error_CV(loss, predictor, X, y, nfolds=5):
    """
    Uses k-fold cross-validation to compute an estimate for the expected prediction error

            Err = E_T[ E_(x,y)[ loss(y, predictor(x) | T] ]

    with (x,y) an unseen test point and T the training set.

    Parameters:
        loss (function): The loss function with signature loss(data, prediction)
        predictor (class providing predict() and fit()): The predictor
        X (array): design matrix (unsplit, i.e. total dataset)
        y (array): response vector (unsplit, i.e. total dataset)
        nfolds (int): Number of folds

    Returns:
        Err (double): estimator for the expected prediction error
        Std(Err) (double): standard error for Err
    """

    def split(X):
        idx = np.arange(N)
        folds = np.array_split(idx, nfolds)
        for i in range(nfolds):
            yield np.hstack(folds[1:]), folds[0]
            folds.append(folds.pop(0))

    N = X.shape[0]
    assert N >= nfolds

    loss_per_fold = np.empty(nfolds)
    for i, (train_idx, test_idx) in enumerate(split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        n = y_test.shape[0]

        loss_per_fold[i] = np.mean(
            loss(y_test, predictor.fit(X_train, y_train).predict(X_test))
        )

    return np.mean(loss_per_fold), np.std(loss_per_fold)


def predicition_error_bootstrap(loss, predictor, X, y, B=50, decomp=True, **kwargs):
    """
    Uses boostrapping to compute an estimate for the expected prediction error

        Err = E_T[ E_(x,y)[ squared_error(y, y' | T] ]
            = E_x[ (E_T[y'] - y)^2 ] + E_x[ E_T[ (E_T[y'] - y')^2 ] ] + noise
            = E_x[ bias^2 ]          + E_x[ variance ]                + noise

    with (x,y) an unseen test point, T the training set and the loss function
    being the squared error. Note that the decomposition is only valid for this
    choice of loss function.

    Parameters:
        loss (function): The loss function with signature loss(data, prediction)
        predictor (class providing predict() and fit()): The predictor
        X (array): design matrix (unsplit, i.e. total dataset)
        y (array): response vector (unsplit, i.e. total dataset)
        B (int): Number of bootstrap samples
        decomp(bool): Compute error decomposition.Sets loss=squared_error
        kwargs (dict): keywords arguments passed to train_test_split()

    Returns:
        prediction_error (double): estimator for the expected prediction error
        If decomp:
            bias_noise: estimate for the expected predictor bias**2 + noise
            variance: estimate for the expected prediction variance
    """
    if decomp:
        loss = squared_error

    X_train, X_test, y_train, y_test = train_test_split(X, y, **kwargs)

    y_pred = np.empty((X_test.shape[0], B))
    bootstrap_indices = np.random.default_rng().integers(
        X_train.shape[0], size=(B, X_train.shape[0])
    )

    for b in range(B):
        X_train_b = X_train[bootstrap_indices[b, :]]
        y_train_b = y_train[bootstrap_indices[b, :]]
        y_pred[:, b] = predictor.fit(X_train_b, y_train_b).predict(X_test)

    y_test = y_test[:, np.newaxis]
    prediction_error = np.mean(np.mean(loss(y_test, y_pred), axis=1, keepdims=True))
    if not decomp:
        return prediction_error

    bias_noise = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True)) ** 2)
    variance = np.mean(np.var(y_pred, axis=1, keepdims=True))

    return prediction_error, bias_noise, variance
