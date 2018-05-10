from sklearn.base import BaseEstimator, RegressorMixin

import numpy as np
from multinomial import MultinomialRegression


class FullDirichletCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.calibrator_ = None

    def fit(self, X, y, *args, **kwargs):
        eps = np.finfo(X.dtype).eps
        X_ = np.log(np.clip(X, eps, 1-eps))

        self.calibrator_ = MultinomialRegression().fit(X_, y, *args, **kwargs)
        self.coef_ = self.calibrator_.coef_
        self.intercept_ = self.calibrator_.intercept_

        return self

    def predict_proba(self, S):
        eps = np.finfo(S.dtype).eps
        S = np.log(np.clip(S, eps, 1-eps))
        return self.calibrator_.predict_proba(S)

    def predict(self, S):
        eps = np.finfo(S.dtype).eps
        S = np.log(np.clip(S, eps, 1-eps))
        return self.calibrator_.predict(S)
