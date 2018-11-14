from __future__ import division

import logging

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import label_binarize
from sklearn.metrics import log_loss

import scipy
import scipy.optimize

from ..utils import clip


class MultinomialRegression(BaseEstimator, RegressorMixin):
    def __init__(self, weights_0=None, method=None, initializer='identity',
                 l2=0.0, comp_l2=False):
        if method not in [None, 'Full', 'FixDiag']:
            raise(ValueError)

        self.weights_ = weights_0
        self.weights_0_ = weights_0
        self.method_ = method
        self.initializer = initializer
        self.l2 = l2
        self.classes = None
        self.comp_l2 = comp_l2  # If true, then Complementary regularization used (off-diagonal regularization)

    def fit(self, X, y, *args, **kwargs):

        X_ = np.hstack((X, np.ones((len(X), 1))))

        self.classes = np.unique(y)

        k = len(self.classes)

        target = label_binarize(y, self.classes)

        if k == 2:
            target = np.hstack([1-target, target])

        n, m = X_.shape

        XXT = (X_.repeat(m, axis=1) * np.hstack([X_]*m)).reshape((n, m, m))

        logging.debug(self.method_)

        self.weights_0_ = self._get_initial_weights(self.initializer)

        if k <= 36:
            weights = _newton_update(self.weights_0_, X_, XXT, target, k,
                                     self.method_, l2=self.l2,
                                     comp_l2=self.comp_l2)
        else:
            res = scipy.optimize.fmin_l_bfgs_b(func=_objective, fprime=_gradient,
                                               x0=self.weights_0_,
                                               args=(X_, XXT, target, k,
                                                     self.method_, self.l2,
                                                     self.comp_l2),
                                               maxls=128,
                                               factr=1.0)
            weights = res[0]

        # logging.debug('===================================================================')
        # if res.success:
        #     logging.debug('optimisation converged!')
        # else:
        #     logging.debug('optimisation not converged!')

        #np.set_printoptions(precision=3)
        #logging.debug('gradient is:')
        #logging.debug(_gradient(weights, X_, XXT, target, k, self.method_))
        #logging.debug('mean target is:')
        #logging.debug(np.mean(target, axis=0))
        #logging.debug('mean output is:')
        #logging.debug(np.mean(_calculate_outputs(_get_weights(weights, k, self.method_), X_), axis=0))
        #logging.debug('obtained paprameters are:')
        #logging.debug(_get_weights(weights, k, self.method_))
        # logging.debug('reason for termination:')
        # logging.debug(res.message)
        #logging.debug('===================================================================')

        self.weights_ = _get_weights(weights, k, self.method_)

        return self

    @property
    def coef_(self):
        return self.weights_[:, :-1]

    @property
    def intercept_(self):
        return self.weights_[:, -1]

    def _get_initial_weights(self, initializer='identity'):
        ''' Returns an array containing only the weights of the full weight
        matrix.

        '''

        if initializer not in ['identity', None, 'preFixDiag']:
            raise ValueError

        k = len(self.classes)

        if self.weights_0_ is None:

            if self.method_ is 'Full':
                if initializer == 'identity':
                    weights_0 = _get_identity_weights(k, self.method_)
                else:
                    weights_0 = np.zeros((k - 1) * (k + 1))

            elif self.method_ is 'FixDiag':
                if initializer == 'identity':
                    weights_0 = _get_identity_weights(k, self.method_)
                else:
                    weights_0 = np.zeros(1)

            if self.method_ is None:
                if initializer == 'identity':
                    weights_0 = _get_identity_weights(k, self.method_)
                else:
                    weights_0 = np.zeros((k - 1) * (k + 1))
        else:
            weights_0 = self.weights_0_

        return weights_0

    def predict_proba(self, S):

        S_ = np.hstack((S, np.ones((len(S), 1))))

        return _calculate_outputs(self.weights_, S_)

    # FIXME Should we change predict for the argmax?
    def predict(self, S):
        return self.predict_proba(S)

    def get_loss_reg(self, X, y):
        # Helper function for saving objective loss and regularization value.

        X_ = np.hstack((X, np.ones((len(X), 1))))  # Add homogeneous coordinates

        weights = self.weights_  # Parameters
        comp_l2 = self.comp_l2
        l2 = self.l2
        k = weights.shape[0]

        # Calculate outputs and loss
        outputs = _calculate_outputs(weights, X_)
        outputs = clip(outputs)
        loss = log_loss(y, outputs, normalize=True)

        #from IPython import embed; embed()
        if l2 != 0:
            if comp_l2:
                temp = weights.copy()
                k = temp.shape[0]
                temp[np.diag_indices(k-1)] = 0  # Set everything on diagonal 0.
                temp[:-1, -2] = 0  # one before last to 0
                #temp[:-1, -1] = 0  # Last column - intercept to 0
                reg = l2*np.sum((temp)**2)
                loss = loss + reg  # EDIT here for complementary matrix
            else:
                reg = l2*np.sum((weights - _get_identity_weights(k, "Full"))**2)
                loss = loss + reg
        else:
            reg = 0
        #logging.debug('Loss is:')
        return (loss, reg)


def _get_identity_weights(n_classes, method):

    weights = None

    if method == 'Full':
        weights = np.zeros((n_classes - 1) * (n_classes + 1))
        weights = _get_weights(weights, n_classes, method)
        weights[np.diag_indices(n_classes - 1)] = 1
        weights[:-1, -2] = -1
        weights = weights[:-1, :].ravel()

    elif method is 'FixDiag':
        weights = np.ones(1)

    elif method is None:
        weights = np.ones((n_classes - 1) * (n_classes + 1))

    return weights


def _newton_update(weights_0, X, XX_T, target, k, method_, maxiter=int(131),
                   ftol=1e-12, gtol=1e-8, l2=0, comp_l2=False):

    L_list = [_objective(weights_0, X, XX_T, target, k, method_, l2, comp_l2)]

    weights = weights_0.copy()

    # TODO move this to the initialization
    if method_ is None:
        weights = np.zeros_like(weights)

    for i in range(0, maxiter):

        gradient = _gradient(weights, X, XX_T, target, k, method_, l2, comp_l2)

        if np.abs(gradient).sum() < gtol:
            break

        hessian = _hessian(weights, X, XX_T, target, k, method_, l2, comp_l2)

        if method_ is 'FixDiag':
            updates = gradient / hessian
        else:
            updates = np.matmul(scipy.linalg.pinv2(hessian), gradient)

        for step_size in np.hstack((np.linspace(1, 0.1, 10),
                                    np.logspace(-2, -32, 31))):

            tmp_w = weights - (updates * step_size).ravel()

            L = _objective(tmp_w, X, XX_T, target, k, method_, l2, comp_l2)

            if (L - L_list[-1]) < 0:
                break

        L_list.append(L)

        logging.debug("{}: after {} iterations log-loss = {:.7e}, sum_grad = {:.7e}".format(
            method_, i, L, np.abs(gradient).sum()))

        if i >= 5:
            if (np.min(np.diff(L_list[-5:])) > -ftol) & (np.sum(np.diff(L_list[-5:]) > 0) == 0):
                weights = tmp_w.copy()
                logging.debug('{}: Terminate as there is not enough changes on Psi.'.format(method_))
                break

        if np.any(np.diff(L_list[-2:]) > 0):
            logging.debug('{}: Terminate as the loss increased {}.'.format(
                method_, np.diff(L_list[-2:])))
            break
        else:
            weights = tmp_w.copy()

    L = _objective(weights, X, XX_T, target, k, method_, l2, comp_l2)
    logging.debug("{}: after {} iterations final log-loss = {:.7e}, sum_grad = {:.7e}".format(
        method_, i, L, np.abs(gradient).sum()))
    #logging.debug("weights = \n{}".format(weights))

    return weights


def _get_weights(params, k, method):
    ''' Reshapes the given params (weights) into the full matrix including 0
    '''

    if method in ['Full', None]:
        weights = np.zeros([k, k+1])
        weights[:-1, :] = params.reshape(-1, k + 1)

    elif method is 'FixDiag':
        weights = np.zeros([k, k])
        weights[np.diag_indices(k - 1)] = params[0]

    return weights


def _get_weights_indices(k, method):
    ''' Returns the indices of the parameters in the full matrix
    '''
    if method in ['Full', None]:
        params = np.arange((k-1)*(k+1)) + 1
        full_matrix = _get_weights(params, k, method)

    elif method is 'FixDiag':
        params = np.arange(k) + 1
        full_matrix = _get_weights(params, k, method)
    return np.where(full_matrix != 0)


def _objective(params, *args):
    (X, _, y, k, method, l2, comp_l2) = args
    weights = _get_weights(params, k, method)
    outputs = _calculate_outputs(weights, X)
    loss = log_loss(y, outputs, normalize=True)
    #from IPython import embed; embed()
    if l2 != 0:
        if comp_l2:  # off-diagonal regularization
            temp = weights.copy()  # TODO - Is copying needed here?
            temp[np.diag_indices(k-1)] = 0  # Set everything on diagonal 0.
            temp[:-1, -2] = 0  # one before last to 0
            reg = l2*np.sum((temp)**2)
            loss = loss + reg
        else:
            reg = l2*np.sum(weights ** 2)
            loss = loss + reg

    #logging.debug('Loss is:')
    #logging.debug(loss)
    #logging.debug('Parameter is:')
    #logging.debug(weights)
    return loss


def _gradient(params, *args):
    (X, _, y, k, method, l2, comp_l2) = args
    weights = _get_weights(params, k, method)
    outputs = _calculate_outputs(weights, X)

    if method in ['Full', None]:

        gradient = np.mean((outputs[:, :-1] - y[:, :-1]).repeat(k+1, axis=1) \
                   * np.hstack([X] * (k-1)), axis=0)

    elif method is 'FixDiag':

        gradient = np.zeros(1)

        gradient[0] += np.mean((outputs[:, :-1] - y[:, :-1]) * X[:, :-1])

    if l2 > 0:
        if comp_l2:
            temp = params.copy()
            temp[0::(k+2)] = 0  # diagonal, +1 for intercept and +1 for a next column.
            temp[(k-1)::(k+1)] = 0  # one before last column
            gradient += 2*l2*(temp)
        else:
            gradient += 2 * l2 * params

    #logging.debug(gradient)

    np.nan_to_num(gradient, copy=False)
    return gradient


def _hessian(params, *args):
    (X, XXT, y, k, method, l2, comp_l2) = args
    weights = _get_weights(params, k, method)
    outputs = _calculate_outputs(weights, X)
    n = np.shape(X)[0]

    if method in ['Full', None]:

        hessian = np.zeros([k ** 2 - 1, k ** 2 - 1])

        for i in range(0, k - 1):
            for j in range(0, k - 1):
                if i <= j:
                    tmp_diff = outputs[:, i] * (int(i == j) - outputs[:, j])
                    tmp_diff = tmp_diff.ravel().repeat((k + 1) ** 2).reshape(n, k + 1, k + 1)
                    hessian[i * (k + 1): (i + 1) * (k + 1), j * (k + 1): (j + 1) * (k + 1)] += np.mean(tmp_diff * XXT,
                                                                                                      axis=0)
                else:
                    hessian[i * (k + 1): (i + 1) * (k + 1), j * (k + 1): (j + 1) * (k + 1)] += \
                            hessian[j * (k + 1): (j + 1) * (k + 1), i * (k + 1): (i + 1) * (k + 1)]

        if comp_l2:  # Regularize hessian for complementary regularization
            for i in range(0, k**2 - 1):  # Go over diagonal indices
                if (i % (k+2) != 0) and (i % (k+1) != k-1):  # Only add on regularized points (so not for diagonal and one before last column)
                    hessian[i, i] += 2*l2

        else:
            hessian[np.diag_indices(k**2 - 1)] += 2*l2

    elif method is 'FixDiag':

        hessian = np.zeros([1])

        hessian[0] = np.mean(np.sum(outputs[:, :-1] * (X[:, :-1] * X[:, :-1]), axis=1)
                            - np.sum(outputs[:, :-1] * X[:, :-1], axis=1) ** 2)

        hessian[0] += 2*l2

    #np.set_printoptions(precision=1)
    #logging.debug('hessian is:')
    #if not (np.all(np.linalg.eigvals(hessian) > 0)):
    #    logging.debug('non-positive-definite Hessian is detected!')
    #logging.debug(hessian)

    np.nan_to_num(hessian, copy=False)

    return hessian


def _calculate_outputs(weights, X):
    mul = np.dot(X, weights.transpose())
    return _softmax(mul)


def _softmax(X):
    """Compute the softmax of matrix X in a numerically stable way."""
    shiftx = X - np.max(X, axis=1).reshape(-1, 1)
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=1).reshape(-1, 1)
