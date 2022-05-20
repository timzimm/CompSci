import numpy as np
from copy import deepcopy


def squared_error(data, prediction):
    return (data - prediction) ** 2


def prediction_error_CV(loss, predictor, X, y, nfolds=5, return_best_predictor=False):
    def split(X):
        idx = np.arange(N)
        folds = np.array_split(idx, nfolds)
        for i in range(nfolds):
            yield np.hstack(folds[1:]), folds[0]
            folds.append(folds.pop(0))

    N = X.shape[0]
    assert N >= nfolds

    loss_per_fold = np.empty(nfolds)

    if return_best_predictor:
        best_trained_predictor = None
        best_loss = np.inf

    for i, (train_idx, test_idx) in enumerate(split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        loss_per_fold[i] = np.mean(
            loss(y_test, predictor.fit(X_train, y_train).predict(X_test))
        )
        if return_best_predictor and loss_per_fold[-1] < best_loss:
            best_trained_predictor = deepcopy(predictor)

    if return_best_predictor:
        return np.mean(loss_per_fold), np.std(loss_per_fold), best_trained_predictor
    else:
        return np.mean(loss_per_fold), np.std(loss_per_fold)
