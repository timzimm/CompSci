import numpy as np

def squared_error(data, prediction):
    return (data - prediction) ** 2 

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
