import numpy as np

from .multinomial import MultinomialRegression
from .fulldirichlet import FullDirichletCalibrator


class FixedDiagonalDirichletCalibrator(FullDirichletCalibrator):
    
    def fit(self, X, y, *args, **kwargs):
        
        eps = np.finfo(X.dtype).eps
        X_ = np.log(np.clip(X, eps, 1-eps))
        k = np.shape(X_)[1]
        for i in range(0, k-1):
            X_[:, i] = X[:, i] - X[:, -1]
        X_ = X_[:, :-1]
        self.calibrator_ = MultinomialRegression(method='FixDiag').fit(X_, y, *args, **kwargs)
        self.coef_ = self.calibrator_.coef_
        self.intercept_ = self.calibrator_.intercept_

        return self

    def predict_proba(self, S):
        
        eps = np.finfo(S.dtype).eps
        S = np.log(np.clip(S, eps, 1-eps))
        k = np.shape(S)[1]
        for i in range(0, k-1):
            S[:, i] = S[:, i] - S[:, -1]
        S = S[:, :-1]

        return self.calibrator_.predict_proba(S)

    def predict(self, S):
        
        eps = np.finfo(S.dtype).eps
        S = np.log(np.clip(S, eps, 1-eps))
        k = np.shape(S)[1]
        for i in range(0, k-1):
            S[:, i] = S[:, i] - S[:, -1]
        S = S[:, :-1]

        return self.calibrator_.predict(S)


