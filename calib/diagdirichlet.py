import numpy as np

from .multinomial import MultinomialRegression
from .fulldirichlet import FullDirichletCalibrator


class DiagonalDirichletCalibrator(FullDirichletCalibrator):
    def fit(self, X, y, *args, **kwargs):
        eps = np.finfo(X.dtype).eps
        X_ = np.log(np.clip(X, eps, 1-eps))

        k = len(np.unique(y))

        weights_0 = np.zeros((k+1, k-1))
        weights_0[np.diag_indices(k-1)] = np.random.rand(k-1)
        weights_0[k-1] = np.random.rand(k-1) * -1
        weights_0[k] = np.random.randn(k-1)

        dims = (k+1, k-1)
        diag_ravel_ind = np.ravel_multi_index(np.diag_indices(k-1), dims)

        k_ind = [np.ones(k-1, dtype=int)*(k-1), np.arange(k-1)]
        k_ravel_ind = np.ravel_multi_index(k_ind, dims)

        intercept_ind = [np.ones(k-1, dtype=int)*k, np.arange(k-1)]
        intercept_ravel_ind = np.ravel_multi_index(intercept_ind, dims)

        bounds = []
        n_params = k**2 - 1
        for ind in range(n_params):
            if ind in diag_ravel_ind:
                bounds.append((0, np.inf))
            elif ind in k_ravel_ind:
                bounds.append((-np.inf, 0))
            elif ind in intercept_ravel_ind:
                bounds.append((-np.inf, np.inf))
            else:
                bounds.append((0, 0))

        self.calibrator_ = MultinomialRegression(
            weights_0=weights_0, bounds=bounds).fit(X_, y, *args, **kwargs)
        self.coef_ = self.calibrator_.coef_
        self.intercept_ = self.calibrator_.intercept_
        return self
