import logging
from sklearn.base import BaseEstimator, RegressorMixin

import numpy as np
from .multinomial import MultinomialRegression
from ..utils import clip_for_log
from sklearn.metrics import log_loss

from .multinomial import _get_identity_weights


class FullDirichletCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self, reg_lambda_list=[0.0], reg_mu_list=[None], weights_init=None,
                 initializer='identity', reg_norm=False, ref_row=True):

        """
        Params:
            weights_init: (nd.array) weights used for initialisation, if None then idendity matrix used. Shape = (n_classes - 1, n_classes + 1)
            comp_l2: (bool) If true, then complementary L2 regularization used (off-diagonal regularization)
        """
        self.reg_lambda_list = reg_lambda_list
        self.reg_mu_list = reg_mu_list  # Complementary L2 regularization. (Off-diagonal regularization)
        self.weights_init = weights_init  # Input weights for initialisation
        self.initializer = initializer
        self.reg_norm = reg_norm
        self.ref_row = ref_row

    def __setup(self):
        self.reg_lambda = 0.0
        self.reg_mu = None
        self.calibrator_ = None
        self.weights_ = self.weights_init

    def fit(self, X, y, X_val=None, y_val=None, *args, **kwargs):

        self.weights_ = self.weights_init

        k = np.shape(X)[1]

        if X_val is None:
            X_val = X.copy()
            y_val = y.copy()

        _X = np.copy(X)
        _X = np.log(clip_for_log(_X))
        _X_val = np.copy(X_val)
        _X_val = np.log(clip_for_log(X_val))

        for i in range(0, len(self.reg_lambda_list)):
            for j in range(0, len(self.reg_mu_list)):
                tmp_cal = MultinomialRegression(method='Full',
                                                reg_lambda=self.reg_lambda_list[i],
                                                reg_mu=self.reg_mu_list[j],
                                                reg_norm=self.reg_norm,
                                                ref_row=self.ref_row)
                tmp_cal.fit(_X, y, *args, **kwargs)
                tmp_loss = log_loss(y_val, tmp_cal.predict_proba(_X_val))

                if (i + j) == 0:
                    final_cal = tmp_cal
                    final_loss = tmp_loss
                    final_reg_lambda = self.reg_lambda_list[i]
                    final_reg_mu = self.reg_mu_list[j]
                elif tmp_loss < final_loss:
                    final_cal = tmp_cal
                    final_loss = tmp_loss
                    final_reg_lambda = self.reg_lambda_list[i]
                    final_reg_mu = self.reg_mu_list[j]

        self.calibrator_ = final_cal
        self.reg_lambda = final_reg_lambda
        self.reg_mu = final_reg_mu
        self.weights_ = self.calibrator_.weights_

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
        _S = np.log(clip_for_log(np.copy(S)))
        return np.asarray(self.calibrator_.predict_proba(_S))

    def predict(self, S):
        _S = np.log(clip_for_log(np.copy(S)))
        return np.asarray(self.calibrator_.predict(_S))
