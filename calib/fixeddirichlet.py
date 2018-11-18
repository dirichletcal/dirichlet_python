import numpy as np

from .multinomial import MultinomialRegression
from .fulldirichlet import FullDirichletCalibrator
from ..utils import clip_for_log


class FixedDiagonalDirichletCalibrator(FullDirichletCalibrator):

    def fit(self, X, y, *args, **kwargs):
        eps = np.finfo(X.dtype).min
        X_ = np.log(clip_for_log(X))
        k = np.shape(X_)[1]
        for i in range(0, k-1):
            X_[:, i] = X[:, i] - X[:, -1]
        X_ = X_[:, :-1]
        self.calibrator_ = MultinomialRegression(method='FixDiag', l2=self.l2).fit(X_, y, *args, **kwargs)
        self.weights_ = self.calibrator_.weights_
        return self

    @property
    def coef_(self):
        return self.calibrator_.coef_

    @property
    def intercept_(self):
        return self.calibrator_.intercept_

    def predict_proba(self, S):
        S = np.log(clip_for_log(S))
        k = np.shape(S)[1]
        for i in range(0, k-1):
            S[:, i] = S[:, i] - S[:, -1]
        S = S[:, :-1]

        return self.calibrator_.predict_proba(S)

    def predict(self, S):
        S = np.log(clip_for_log(S))
        k = np.shape(S)[1]
        for i in range(0, k-1):
            S[:, i] = S[:, i] - S[:, -1]
        S = S[:, :-1]

        return self.calibrator_.predict(S)


