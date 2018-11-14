import logging
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import confusion_matrix

import numpy as np
from .multinomial import MultinomialRegression
from ..utils import clip_for_log
from sklearn.metrics import log_loss

from .multinomial import _get_identity_weights

class ConfDirichletCalibrator(BaseEstimator, RegressorMixin):
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

        y_pred = X.argmax(axis=1)
        conf_matrix = confusion_matrix(y, y_pred)
        c = conf_matrix - conf_matrix[-1]
        c = 2 * ((c - c.min()) / (c.max() - c.min())) - 1.0
        self.calibrator_ = {'coef_': c}
        return self

    @property
    def coef_(self):
        return self.calibrator_['coef_']

    @property
    def intercept_(self):
        return np.zeros(len(self.calibrator_['coef_']))

    def predict_proba(self, S):
        S = np.log(clip_for_log(S))

        return _calculate_outputs(self.calibrator_['coef_'], S)

    def predict(self, S):
        probas = self.predict_proba(S)
        return probas.argmax(axis=1)
        
def _calculate_outputs(weights, X):
    mul = np.dot(X, weights.transpose())
    return _softmax(mul)


def _softmax(X):
    """Compute the softmax of matrix X in a numerically stable way."""
    shiftx = X - np.max(X, axis=1).reshape(-1, 1)
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=1).reshape(-1, 1)