from __future__ import division

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize


class _DiagonalDirichletCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.matrix_ = None
        self.intercepts_ = None

    def fit(self, X, y, *args, **kwargs):
        eps = np.finfo(X.dtype).eps
        X = np.log(np.clip(X, eps, 1-eps))

        k = len(np.unique(y))
        matrix = np.zeros((k-1, k))
        matrix[np.diag_indices(k-1)] = np.random.randn(k-1)
        matrix[:, k-1] = np.random.randn(k-1)

        intercepts = np.random.randn(k-1)

        target = label_binarize(y, range(k))


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
        ).fit(X, y, *args, **kwargs)

        return self

    def predict_proba(self, S):
        eps = np.finfo(S.dtype).eps
        S = np.log(np.clip(S, eps, 1-eps))
        return self.calibrator_.predict_proba(S)

    def predict(self, S):
        eps = np.finfo(S.dtype).eps
        S = np.log(np.clip(S, eps, 1-eps))
        return self.calibrator_.predict(S)

    # def alphas(self):
# a11-a13 a21-a23 a31-a33
# a12-a13 a22-a23 a32-a33
