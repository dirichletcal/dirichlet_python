import logging
from sklearn.base import BaseEstimator
from sklearn.preprocessing import label_binarize
import numpy
import scipy.special
from ..utils import clip_for_log



class GenerativeDirichletCalibrator(BaseEstimator):

    def __init__(self, alpha_init=None):

        self.calibrator_ = None

        self.alpha = alpha_init

        self.classes = None

        self.weights_ = None

    def fit(self, X, y, X_val=None, y_val=None):
        
        ln_X = numpy.log(clip_for_log(X))

        self.classes = numpy.unique(y)

        k = len(self.classes)

        self.weights_ = numpy.zeros((k, k+1))

        target = label_binarize(y, self.classes)

        if k == 2:
            target = numpy.hstack([1-target, target])

        pi = numpy.mean(target, axis=0)

        if self.alpha is None:
            
            self.alpha = numpy.ones((k, k))

        for i in range(0, k):
            
            self.alpha[i, :] = _fit_dirichlet(ln_X[target[:, i] == 1], self.alpha[i, :])

            self.weights_[i, :-1] = self.alpha[i, :] - 1

            self.weights_[i, -1] = numpy.log(pi[i]) \
                    + scipy.special.gammaln(numpy.sum(self.alpha[i, :])) \
                    - numpy.sum(scipy.special.gammaln(self.alpha[i, :]))
        
        return self

    def predict_proba(self, S):
        S_ = numpy.hstack((numpy.log(clip_for_log(S)), numpy.ones((len(S), 1))))

        return _calculate_outputs(self.weights_, S_)

    def predict(self, S):
        return self.predict_proba(S)


def _fit_dirichlet(ln_X, alpha):

    gamma = -scipy.special.digamma(1)

    k = numpy.shape(alpha)[0]

    for i in range(0, 1000):

        alpha_old = alpha.copy()

        Psi = scipy.special.digamma(numpy.sum(alpha)) + numpy.mean(ln_X, axis=0)

        for l in range(0, k):

            if Psi[l] >= - 2.22:
                alpha[l] = numpy.exp(Psi[l]) + 0.5
            else:
                alpha[l] = - 1 / (Psi[l] + gamma)

            for j in range(0, 5):

                alpha[l] = alpha[l] - \
                              (scipy.special.digamma(alpha[l]) - Psi[l]) / scipy.special.polygamma(1, alpha[l])

        if numpy.sum(numpy.abs(alpha - alpha_old)) <= 1e-15:
            break

    return alpha


def _calculate_outputs(weights_, X):
    mul = numpy.dot(X, weights_.transpose())
    return _softmax(mul)


def _softmax(X):
    """Compute the softmax of matrix X in a numerically stable way."""
    shiftx = X - numpy.max(X, axis=1).reshape(-1, 1)
    exps = numpy.exp(shiftx)
    return exps / numpy.sum(exps, axis=1).reshape(-1, 1)

