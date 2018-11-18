import logging
from sklearn.base import BaseEstimator, RegressorMixin

import numpy as np
from .multinomial import MultinomialRegression
from ..utils import clip_for_log
from sklearn.metrics import log_loss

from .multinomial import _get_identity_weights

class FullDirichletCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self, l2=0.0, comp_l2=False, weights_init=None,
                 initializer='identity'):
    
        """
        Params:
            weights_init: (nd.array) weights used for initialisation, if None then idendity matrix used. Shape = (n_classes - 1, n_classes + 1)
            l2: (float) regularization parameter (lambda)
            comp_l2: (bool) If true, then complementary L2 regularization used (off-diagonal regularization)
        """
    
        self.calibrator_ = None
        self.weights_ = weights_init  # Input weights for initialisation
        self.l2 = l2
        self.comp_l2 = comp_l2  # Complementary L2 regularization. (Off-diagonal regularization)
        self.initializer = initializer
        
    def fit(self, X, y, X_val=None, y_val=None, *args, **kwargs):

        _X = np.copy(X)
        _X = np.log(clip_for_log(_X))

        if isinstance(self.l2, list):
            _X_val = np.copy(X_val)
            if len(_X_val.shape) == 1:
                _X_val = np.vstack(((1-_X_val), _X_val)).T
            _X_val = np.log(clip_for_log(_X_val))
            cal_list = []
            losses = []
            for l2 in self.l2:
                if self.initializer == 'preFixDiag':
                    from .fixeddirichlet import FixedDiagonalDirichletCalibrator
                    calibrator = FixedDiagonalDirichletCalibrator(initializer='identity')
                    calibrator.fit(X, y, *args, **kwargs)
                    self.weights_ = _get_identity_weights(len(calibrator.calibrator_.classes), 'Full')
                    self.weights_ *= calibrator.calibrator_.weights_[0,0]
                calibrator = MultinomialRegression(method='Full', l2=l2,
                                                   comp_l2=self.comp_l2,
                                                   weights_0=self.weights_)
                calibrator.fit(_X, y, *args, **kwargs)
                cal_list.append(calibrator)
                losses.append(log_loss(y_val, cal_list[-1].predict_proba(_X_val)))

            lower_loss_index = np.argmin(losses)
            logging.debug('Best l2 regularization = {}'.format(self.l2[lower_loss_index]))
            self.calibrator_ = cal_list[lower_loss_index]
            self.l2 = self.l2[lower_loss_index]
        else:
            if self.initializer == 'preFixDiag':
                from .fixeddirichlet import FixedDiagonalDirichletCalibrator
                calibrator = FixedDiagonalDirichletCalibrator(initializer='identity')
                calibrator.fit(X, y, *args, **kwargs)
                self.weights_ = _get_identity_weights(len(calibrator.calibrator_.classes), 'Full')
                self.weights_ *= calibrator.calibrator_.weights_[0,0]
            self.calibrator_ = MultinomialRegression(method='Full', l2=self.l2,
                                                     comp_l2=self.comp_l2,
                                                     weights_0=self.weights_)
            self.calibrator_.fit(_X, y, *args, **kwargs)

        self.weights_ = self.calibrator_.weights_
        return self

    @property
    def coef_(self):
        return self.calibrator_.coef_

    @property
    def intercept_(self):
        return self.calibrator_.intercept_

    def predict_proba(self, S):
        S = np.log(clip_for_log(S))
        return self.calibrator_.predict_proba(S)

    def predict(self, S):
        S = np.log(clip_for_log(S))
        return self.calibrator_.predict(S)
    
    def get_loss_reg(self, X, y):
        # Helper function for getting objective loss and regularization value.
        X = np.log(clip_for_log(X))
        return self.calibrator_.get_loss_reg(X, y)
        
