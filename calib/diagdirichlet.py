import numpy as np

from .multinomial import MultinomialRegression
from .fulldirichlet import FullDirichletCalibrator


class DiagonalDirichletCalibrator(FullDirichletCalibrator):
    
     def fit(self, X, y, *args, **kwargs):
        eps = np.finfo(X.dtype).eps
        X_ = np.log(np.clip(X, eps, 1-eps))
        self.calibrator_ = MultinomialRegression(method='Diag').fit(X_, y, *args, **kwargs)
        self.coef_ = self.calibrator_.coef_
        self.intercept_ = self.calibrator_.intercept_

        return self
