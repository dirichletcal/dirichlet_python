import logging

from sklearn.base import BaseEstimator, RegressorMixin

from dirichlet.calib.fulldirichlet import FullDirichletCalibrator
from dirichlet.calib.diagdirichlet import DiagonalDirichletCalibrator
from dirichlet.calib.fixeddirichlet import FixedDiagonalDirichletCalibrator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DirichletCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self, matrix_type='full'):
        self.matrix_type = matrix_type

    def fit(self, X, y, *args, **kwargs):

        if self.matrix_type == 'diagonal':
            self.calibrator_ = DiagonalDirichletCalibrator()
        elif self.matrix_type == 'fixed_diagonal':
            self.calibrator_ = FixedDiagonalDirichletCalibrator()
        else:
            self.calibrator_ = FullDirichletCalibrator()

        self.calibrator_ = self.calibrator_.fit(X, y, *args, **kwargs)
        self.coef_ = self.calibrator_.coef_
        self.intercept_ = self.calibrator_.intercept_
        return self

    def predict_proba(self, S):
        return self.calibrator_.predict_proba(S)

    def predict(self, S):
        return self.calibrator_.predict(S)
