import numpy as np
from metric import mse, squared_error
from sklearn.model_selection import train_test_split


def prediction_error_CV(loss, predictor, X, y, nfolds=5):
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

        loss_per_fold[i] = np.mean(
            loss(y_test, predictor.fit(X_train, y_train).predict(X_test))
        )

    return np.mean(loss_per_fold), np.std(loss_per_fold)


def predicition_error_bootstrap(loss, predictor, X, y, B=50, decomp=True, **kwargs):
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
