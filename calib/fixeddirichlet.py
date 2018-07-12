import numpy as np

from .multinomial import MultinomialRegression
from .fulldirichlet import FullDirichletCalibrator


class FixedDiagonalDirichletCalibrator(FullDirichletCalibrator):
    def fit(self, X, y, *args, **kwargs):
        eps = np.finfo(X.dtype).eps
        X_ = np.log(np.clip(X, eps, 1-eps))

        k = len(np.unique(y))

        weights_0 = np.append(np.random.randn(k-1), np.random.rand())
        bounds = [(-np.inf, np.inf) for _ in range(k-1)]
        bounds.append((0, np.inf))

        self.calibrator_ = MultinomialRegression(
            weights_0=weights_0, bounds=bounds).fit(X_, y, *args, **kwargs)
        self.coef_ = self.calibrator_.coef_
        self.intercept_ = self.calibrator_.intercept_
        return self
