from .version import __version__
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

from .calib.fulldirichlet import FullDirichletCalibrator
from .calib.diagdirichlet import DiagonalDirichletCalibrator
from .calib.fixeddirichlet import FixedDiagonalDirichletCalibrator
from .calib.gendirichlet import GenerativeDirichletCalibrator


class DirichletCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self, matrix_type='full', l2=0.0, comp_l2=False,
                 initializer='identity'):
        if matrix_type not in ['full', 'full_gen', 'diagonal',
                               'fixed_diagonal']:
            raise(ValueError)

        self.matrix_type = matrix_type
        self.l2 = l2
        if isinstance(l2, list):
            self.l2_grid = l2
        else:
            self.l2_grid = [l2]
        if isinstance(comp_l2, list):
            self.comp_l2 = comp_l2
        else:
            self.comp_l2 = [comp_l2]
        self.initializer = initializer

    def fit(self, X, y, X_val=None, y_val=None, **kwargs):

        if self.matrix_type == 'diagonal':
            self.calibrator_ = DiagonalDirichletCalibrator(
                l2=self.l2, initializer=self.initializer)
        elif self.matrix_type == 'fixed_diagonal':
            self.calibrator_ = FixedDiagonalDirichletCalibrator(
                l2=self.l2, initializer=self.initializer)
        elif self.matrix_type == 'full':
            self.calibrator_ = FullDirichletCalibrator(
                reg_lambda_list=self.l2_grid, reg_mu_list=self.comp_l2,
                initializer=self.initializer)
        elif self.matrix_type == 'full_gen':
            self.calibrator_ = GenerativeDirichletCalibrator()
        else:
            raise(ValueError)

        _X = np.copy(X)
        if len(X.shape) == 1:
            _X = np.vstack(((1-_X), _X)).T

        _X_val = X_val
        if X_val is not None:
            _X_val = np.copy(X_val)
            if len(X_val.shape) == 1:
                _X_val = np.vstack(((1-_X_val), _X_val)).T

        self.calibrator_ = self.calibrator_.fit(_X, y, X_val=_X_val,
                                                y_val=y_val, **kwargs)

        if hasattr(self.calibrator_, 'l2'):
            self.l2 = self.calibrator_.l2
        if hasattr(self.calibrator_, 'weights_'):
            self.weights_ = self.calibrator_.weights_
        if hasattr(self.calibrator_, 'coef_'):
            self.coef_ = self.calibrator_.coef_
        if hasattr(self.calibrator_, 'intercept_'):
            self.intercept_ = self.calibrator_.intercept_
        return self

    @property
    def cannonical_weights(self):
        b = self.weights_[:, -1]
        W = self.weights_[:, :-1]
        col_min = np.min(W, axis=0)
        A = W - col_min
        def softmax(z):
            return np.divide(np.exp(z), np.sum(np.exp(z)))
        c = softmax(np.matmul(W, np.log(np.ones(len(b))/len(b))) + b)
        return np.hstack((A, c.reshape(-1, 1)))

    def predict_proba(self, S):

        _S = np.copy(S)
        if len(S.shape) == 1:
            _S = np.vstack(((1-_S), _S)).T
            return self.calibrator_.predict_proba(_S)[:, 1]

        return self.calibrator_.predict_proba(_S)

    def predict(self, S):

        _S = np.copy(S)
        if len(S.shape) == 1:
            _S = np.vstack(((1-_S), _S)).T
            return self.calibrator_.predict(_S)[:, 1]

        return self.calibrator_.predict(_S)
