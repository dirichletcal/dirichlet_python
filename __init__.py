import logging

from sklearn.base import BaseEstimator, RegressorMixin

from dirichlet.calib.fulldirichlet import FullDirichletCalibrator
from dirichlet.calib.diagdirichlet import DiagonalDirichletCalibrator
from dirichlet.calib.fixeddirichlet import FixedDiagonalDirichletCalibrator


class DirichletCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self, matrix_type='full'):
        if matrix_type not in ['full', 'diagonal', 'fixed_diagonal']:
            raise(ValueError)

        self.matrix_type = matrix_type

    def fit(self, X, y, *args, **kwargs):

        if self.matrix_type == 'diagonal':
            self.calibrator_ = DiagonalDirichletCalibrator()
        elif self.matrix_type == 'fixed_diagonal':
            self.calibrator_ = FixedDiagonalDirichletCalibrator()
        else:
            self.calibrator_ = FullDirichletCalibrator()

        self.calibrator_ = self.calibrator_.fit(X, y, *args, **kwargs)
        return self


    @property
    def coef_(self):
        return self.calibrator_.coef_


    @property
    def intercept_(self):
        return self.calibrator_.intercept_


    def predict_proba(self, S):
        return self.calibrator_.predict_proba(S)

    def predict(self, S):
        return self.calibrator_.predict(S)
