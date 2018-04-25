from __future__ import division

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression


class DirichletCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self):
            self.calibrator_ = None

    def fit(self, X, y, sample_weight=None):
        n = len(y)

        self.calibrator_ = LogisticRegression(
            C=99999999999,
            multi_class='multinomial', solver='saga'
            ).fit(
                np.log(X), 
                y, 
                sample_weight
                )

        return self

    def predict_proba(self, S):
        return self.calibrator_.predict_proba(S)

    def predict(self, S):
        return self.calibrator_.predict(S)
