import numpy as np

from .multinomial import MultinomialRegression
from .fulldirichlet import FullDirichletCalibrator
from ..utils import clip_for_log


class DiagonalDirichletCalibrator(FullDirichletCalibrator):

     def fit(self, X, y, *args, **kwargs):
        X_ = np.log(clip_for_log(X))
        self.calibrator_ = MultinomialRegression(method='Diag').fit(X_, y, *args, **kwargs)
        self.coef_ = self.calibrator_.coef_
        self.intercept_ = self.calibrator_.intercept_

        return self
