from sklearn.base import BaseEstimator, RegressorMixin

import numpy as np
from .multinomial import MultinomialRegression
from ..utils import clip_for_log


class FullDirichletCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self, l2=0.0):
        self.calibrator_ = None
        self.l2 = l2

    def fit(self, X, y, *args, **kwargs):
        X_ = np.log(clip_for_log(X))
        self.calibrator_ = MultinomialRegression(method='Full', l2=self.l2).fit(X_, y, *args, **kwargs)
        return self

    @property
    def coef_(self):
        return self.calibrator_.coef_

    @property
    def intercept_(self):
        return self.calibrator_.intercept_

    def predict_proba(self, S):
        S = np.log(clip_for_log(S))
        return self.calibrator_.predict_proba(S)

    def predict(self, S):
        S = np.log(clip_for_log(S))
        return self.calibrator_.predict(S)
