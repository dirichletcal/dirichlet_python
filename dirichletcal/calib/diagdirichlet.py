import numpy as np
from .multinomial import MultinomialRegression
from .fulldirichlet import FullDirichletCalibrator
from ..utils import clip_for_log
from sklearn.metrics import log_loss


class DiagonalDirichletCalibrator(FullDirichletCalibrator):
    def fit(self, X, y, X_val=None, y_val=None, *args, **kwargs):

        self.weights_ = self.weights_init

        if X_val is None:
            X_val = X.copy()
            y_val = y.copy()

        _X = np.copy(X)
        _X = np.log(clip_for_log(_X))
        _X_val = np.copy(X_val)
        _X_val = np.log(clip_for_log(X_val))

        self.calibrator_ = MultinomialRegression(
            method='Diag', reg_lambda=self.reg_lambda, reg_mu=self.reg_mu,
            reg_norm=self.reg_norm, ref_row=self.ref_row,
            optimizer=self.optimizer)
        self.calibrator_.fit(_X, y, *args, **kwargs)
        self.final_loss = log_loss(y_val, self.calibrator_.predict_proba(_X_val))

        return self
