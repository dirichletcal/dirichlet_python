import logging
from sklearn.base import BaseEstimator, RegressorMixin

import numpy as np
from .multinomial import MultinomialRegression
from ..utils import clip_for_log
from sklearn.metrics import log_loss

from .multinomial import _get_identity_weights


class FullDirichletCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self, reg_lambda=0.0, reg_mu=None, weights_init=None,
                 initializer='identity', reg_norm=False, ref_row=True):

        """
        Params:
            weights_init: (nd.array) weights used for initialisation, if None then idendity matrix used. Shape = (n_classes - 1, n_classes + 1)
            comp_l2: (bool) If true, then complementary L2 regularization used (off-diagonal regularization)
        """

        self.calibrator_ = None
        self.weights_ = weights_init  # Input weights for initialisation
        self.reg_mu = None
        self.reg_lambda = reg_lambda
        self.reg_mu = reg_mu  # Complementary L2 regularization. (Off-diagonal regularization)
        self.initializer = initializer
        self.reg_norm = reg_norm
        self.ref_row = ref_row

    def fit(self, X, y, X_val=None, y_val=None, *args, **kwargs):

        k = np.shape(X)[1]

        if X_val is None:
            X_val = X.copy()
            y_val = y.copy()

        _X = np.copy(X)
        _X = np.log(clip_for_log(_X))
        _X_val = np.copy(X_val)
        _X_val = np.log(clip_for_log(X_val))

        self.calibrator_ = MultinomialRegression(method='Full',
                                        reg_lambda=self.reg_lambda,
                                        reg_mu=self.reg_mu,
                                        reg_norm=self.reg_norm,
                                        ref_row=self.ref_row)
        self.calibrator_.fit(_X, y, *args, **kwargs)
        final_loss = log_loss(y_val, self.calibrator_.predict_proba(_X_val))

        return self

    @property
    def weights(self):
        if self.calibrator_ is not None:
            return self.calibrator_.weights_
        return self.weights_init

    @property
    def coef_(self):
        return self.calibrator_.coef_

    @property
    def intercept_(self):
        return self.calibrator_.intercept_

    def predict_proba(self, S):
        S = np.log(clip_for_log(S))
        return np.asarray(self.calibrator_.predict_proba(S))

    def predict(self, S):
        S = np.log(clip_for_log(S))
        return np.asarray(self.calibrator_.predict(S))
