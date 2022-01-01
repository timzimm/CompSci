from transformation import Transformation
from linear_model import RegressionBase
from linear_model import ClassificationBase


class Pipeline(RegressionBase):
    def __init__(self, steps):
        for step in steps[:-1]:
            if not isinstance(step, Transformation):
                raise TypeError("All intermediate steps should be Transformations.")
        if not (
            isinstance(steps[-1], RegressionBase)
            or isinstance(steps[-1], ClassificationBase)
        ):
            raise TypeError("Final step must be RegressionBase or ClassificationBase")
        self.steps = steps

    def __fit_transform(self, X):
        X_pipeline = self.steps[0].fit_transform(X)
        for step in self.steps[1:-1]:
            X_pipeline = step.fit_transform(X_pipeline)
        return X_pipeline

    def __transform(self, X):
        X_pipeline = self.steps[0].transform(X)
        for step in self.steps[1:-1]:
            X_pipeline = step.transform(X_pipeline)
        return X_pipeline

    def fit(self, X, y):
        self.steps[-1].fit(self.__fit_transform(X), y)
        return self

    def predict(self, X):
        return self.steps[-1].predict(self.__transform(X))
