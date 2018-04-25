from __future__ import division

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression


class DirichletCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self):
            self.calibrator_ = None

    def fit(self, X, y, *args, **kwargs):
        n = len(y)

        eps = np.finfo(X.dtype).eps
        X = np.log(np.clip(X, eps, 1-eps))

        self.calibrator_ = LogisticRegression(
            C=99999999999,
            multi_class='multinomial', solver='saga'
            ).fit( X, y, *args, **kwargs)

        return self

    def predict_proba(self, S):
        eps = np.finfo(S.dtype).eps
        S = np.log(np.clip(S, eps, 1-eps))
        return self.calibrator_.predict_proba(S)

    def predict(self, S):
        eps = np.finfo(S.dtype).eps
        S = np.log(np.clip(S, eps, 1-eps))
        return self.calibrator_.predict(S)
