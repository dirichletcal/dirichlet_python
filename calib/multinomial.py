from __future__ import division

import logging

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import label_binarize
from sklearn.metrics import log_loss

from scipy.optimize import minimize

import scipy
import scipy.optimize

from ..utils import clip


class MultinomialRegression(BaseEstimator, RegressorMixin):
    def __init__(self, weights_0=None, method='Full', initializer='identity',
                 l2=0.0):
        if method not in ['Full', 'Diag', 'FixDiag']:
            raise(ValueError)

        self.weights_ = weights_0
        self.weights_0_ = weights_0
        self.method_ = method
        self.initializer = initializer
        self.l2 = l2

    def fit(self, X, y, *args, **kwargs):
        X_ = np.hstack((X, np.ones((len(X), 1))))

        self.classes = np.unique(y)

        k = len(self.classes)
        target = label_binarize(y, self.classes)
        if k == 2:
            target = np.hstack([1-target, target])

        n, m = X_.shape

        XXT = np.zeros([n, m, m])

        for i in range(0, n):
            XXT[i, :, :] = np.matmul(X_[i, :].reshape(-1, 1), X_[i, :].reshape(-1, 1).transpose())

        logging.debug(self.method_)

        self.weights_0_ = self._get_initial_weights(self.initializer)

        weights = _newton_update(self.weights_0_, X_, XXT, target, k,
                                 self.method_, l2=self.l2)

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
        if initializer not in ['identity', 'randn']:
            raise(ValueError)

        k = len(self.classes)

        if self.weights_0_ is None:

            if self.method_ is 'Full':
                if initializer == 'identity':
                    weights_0 = _get_identity_weights(k, self.method_)
                else:
                    weights_0 = np.random.randn((k - 1) * (k + 1))
                    weights_0 = _get_weights(weights_0, k, self.method_)
                    weights_0[np.diag_indices(k - 1)] = np.abs(weights_0[np.diag_indices(k - 1)])
                    weights_0[:, k - 1] = np.abs(weights_0[:, k - 1]) * -1

                weights_0 = weights_0[_get_weights_indices(k, self.method_)]

            elif self.method_ is 'Diag':
                if initializer == 'identity':
                    weights_0 = _get_identity_weights(k, self.method_)
                else:
                    weights_0 = np.random.randn(k + (k - 1))
                    weights_0 = _get_weights(weights_0, k, self.method_)
                    weights_0[np.diag_indices(k)] = np.abs(weights_0[np.diag_indices(k)])

                weights_0 = weights_0[_get_weights_indices(k, self.method_)]

            elif self.method_ is 'FixDiag':
                if initializer == 'identity':
                    weights_0 = _get_identity_weights(k, self.method_)
                else:
                    weights_0 = np.random.randn(k)
                    weights_0[0] = np.abs(weights_0[0])

        else:
            weights_0 = self.weights_0_

        return weights_0


    def predict_proba(self, S):
        S_ = np.hstack((S, np.ones((len(S), 1))))
        return _calculate_outputs(self.weights_, S_)

    # FIXME Should we change predict for the argmax?
    def predict(self, S):
        return self.predict_proba(S)


def _get_identity_weights(n_classes, method):
    if method == 'Full':
        weights = np.zeros((n_classes - 1) * (n_classes + 1))
        weights = _get_weights(weights, n_classes, method)
        weights[np.diag_indices(n_classes - 1)] = 1
        weights[:-1, -2] = -1
    elif method is 'Diag':
        weights = np.zeros((n_classes, n_classes+1))
        weights[np.diag_indices(n_classes)] = 1
    elif method is 'FixDiag':
        weights = np.zeros(n_classes)
        weights[0] = 1
    return weights


def _newton_update(weights_0, X, XX_T, target, k, method_, maxiter=int(1e3),
                   ftol=1e-12, gtol=1e-12, l2=0):

    L_list = [_objective(weights_0, X, XX_T, target, k, method_, l2)]

    weights = weights_0.copy()

    for i in range(0, maxiter):

        gradient = _gradient(weights, X, XX_T, target, k, method_, l2)

        if np.abs(gradient).sum() < gtol:
            break

        hessian = _hessian(weights, X, XX_T, target, k, method_, l2)

        for step_size in np.hstack((np.linspace(1, 0.1, 10),
                                    np.linspace(0.09, 0.01, 9),
                                    np.linspace(0.009, 0.001, 9),
                                    np.linspace(0.0009, 0.0001, 9),
                                    np.linspace(0.00009, 0.00001, 9),
                                    np.linspace(0.000009, 0.000001, 9),
                                    1e-8, 1e-16, 1e-32)):

            updates = (np.matmul(scipy.linalg.pinv2(hessian), gradient.reshape(-1, 1)) * step_size).ravel()

            tmp_w = weights - updates

            L = _objective(tmp_w, X, XX_T, target, k, method_, l2)

            if (L - L_list[-1]) < 0:
                break

        L_list.append(L)

        logging.debug("{}: after {} iterations log-loss = {:.2e}, sum_grad = {:.2e}".format(
            method_, i, L, np.abs(gradient).sum()))

        if i >= 5:
            if (np.min(np.diff(L_list[-5:])) > -ftol) & (np.sum(np.diff(L_list[-5:]) > 0) == 0):
                weights = tmp_w.copy()
                logging.debug('{}: Terminate as there is not enough changes on Psi.'.format(method_))
                break

        if np.any(np.diff(L_list[-2:]) > 0):
            logging.debug('{}: Terminate as the loss increased {:.2e}.'.format(
                method_, np.diff(L_list[-2:])))
            break
        else:
            weights = tmp_w.copy()

    if 'L' not in locals():
        L = _objective(weights, X, XX_T, target, k, method_, l2)
    logging.debug("{}: after {} iterations final log-loss = {:.2e}, sum_grad = {:.2e}".format(
        method_, i, L, np.abs(gradient).sum()))

    return weights


def _get_weights(params, k, method):
    ''' Reshapes the given params (weights) into the full matrix including 0
    '''

    if method is 'Full':
        weights = np.zeros([k, k+1])
        weights[:-1, :] = params.reshape(-1, k + 1)

    elif method is 'Diag':
        weights = np.zeros([k, k+1])
        tmp_params = params[:-1].reshape(-1, 2)
        weights[np.diag_indices(k - 1)] = tmp_params[:, 0]
        weights[:-1, -1] = tmp_params[:, 1]
        weights[-1, k - 1] = params[-1]

    elif method is 'FixDiag':
        weights = np.zeros([k, k])
        weights[np.diag_indices(k - 1)] = params[0]
        weights[:-1, -1] = params[1:]

    return weights


def _get_weights_indices(k, method):
    ''' Returns the indices of the parameters in the full matrix
    '''
    if method is 'Full':
        params = np.arange((k-1)*(k+1)) + 1
        full_matrix = _get_weights(params, k, method)
    elif method is 'Diag':
        params = np.arange(k + k - 1) + 1
        full_matrix = _get_weights(params, k, method)
    elif method is 'FixDiag':
        params = np.arange(k) + 1
        full_matrix = _get_weights(params, k, method)
    return np.where(full_matrix != 0)


def _objective(params, *args):
    (X, _, y, k, method, l2) = args
    weights = _get_weights(params, k, method)
    outputs = _calculate_outputs(weights, X)
    loss = log_loss(y, outputs)
    #from IPython import embed; embed()
    if l2 != 0:
        loss = loss + l2*np.sum((weights - _get_identity_weights(k, method))**2)
    #logging.debug('Loss is:')
    #logging.debug(loss)
    #logging.debug('Parameter is:')
    #logging.debug(weights)
    return loss


def _gradient(params, *args):
    (X, _, y, k, method, l2) = args
    weights = _get_weights(params, k, method)
    outputs = _calculate_outputs(weights, X)

    if method is 'Full':

        gradient = np.zeros((k - 1) * (k + 1))

        for i in range(0, k - 1):

            gradient[i * (k + 1):(i + 1) * (k + 1)] += \
                    np.sum((outputs[:, i] - y[:, i]).reshape(-1, 1).repeat(k+1, axis=1) * X, axis=0)

        if l2 > 0:
            gradient += 2*l2*(params - _get_identity_weights(k,
                                                             method)[_get_weights_indices(k,
                                                                                          method)])
    elif method is 'Diag':

        gradient = np.zeros(k + (k - 1))

        for i in range(0, k - 1):

            gradient[i * 2:(i + 1) * 2] += \
                    np.sum((outputs[:, i] - y[:, i]).reshape(-1, 1).repeat(2, axis=1) * X[:, [i, k]], axis=0)

        gradient[-1] = np.sum((outputs[:, i] - y[:, i]) * X[:, k])

    elif method is 'FixDiag':

        gradient = np.zeros(k)

        gradient[0] += np.sum((outputs[:, :-1] - y[:, :-1]) * X[:, :-1])

        for i in range(0, k - 1):

            gradient[i + 1] += np.sum((outputs[:, i] - y[:, i]) * X[:, k - 1])

    #logging.debug(gradient)
    np.nan_to_num(gradient, copy=False)
    return gradient


def _hessian(params, *args):
    (X, XXT, y, k, method, l2) = args
    weights = _get_weights(params, k, method)
    outputs = _calculate_outputs(weights, X)
    n = np.shape(X)[0]

    if method is 'Full':

        hessian = np.zeros([k ** 2 - 1, k ** 2 - 1])

        for i in range(0, k - 1):
            for j in range(0, k - 1):
                if i <= j:
                    tmp_diff = outputs[:, i] * (int(i == j) - outputs[:, j])
                    tmp_diff = tmp_diff.ravel().repeat((k + 1) ** 2).reshape(n, k + 1, k + 1)
                    hessian[i * (k + 1): (i + 1) * (k + 1), j * (k + 1): (j + 1) * (k + 1)] += np.sum(tmp_diff * XXT, axis=0)
                else:
                    hessian[i * (k + 1): (i + 1) * (k + 1), j * (k + 1): (j + 1) * (k + 1)] += \
                            hessian[j * (k + 1): (j + 1) * (k + 1), i * (k + 1): (i + 1) * (k + 1)]
        hessian[np.diag_indices(k**2 - 1)] += 2*l2

    elif method is 'Diag':

        hessian = np.zeros([2*k - 1, 2*k - 1])

        for i in range(0, k - 1):
            for j in range(0, k - 1):
                if i <= j:
                    sub_XXT_1 = XXT[:, [i, i, k, k], [i, k, i, k]].reshape(-1, 2, 2)
                    sub_XXT_2 = XXT[:, [j, j, k, k], [i, k, i, k]].reshape(-1, 2, 2)

                    tmp_product = (outputs[:, i] * outputs[:, j]).ravel().repeat(4).reshape(n, 2, 2)

                    if i == j:
                        tmp_mu = outputs[:, i].ravel().repeat(4).reshape(n, 2, 2)
                        hessian[i * 2: (i + 1) * 2, j * 2: (j + 1) * 2] = np.sum(tmp_mu * sub_XXT_1 - tmp_product * sub_XXT_2, axis=0)
                    else:
                        hessian[i * 2: (i + 1) * 2, j * 2: (j + 1) * 2] = np.sum(-tmp_product * sub_XXT_2, axis=0)
                else:
                        hessian[i * 2: (i + 1) * 2, j * 2: (j + 1) * 2] = hessian[j * 2: (j + 1) * 2, i * 2: (i + 1) * 2]

        for i in range(0, k - 1):
            sub_XXT = XXT[:, [k - 1, k - 1], [i, k]].reshape(-1, 2, 1)
            tmp_product = (outputs[:, i] * outputs[:, -1]).ravel().repeat(2).reshape(n, 2, 1)
            hessian[i * 2: (i + 1) * 2, -1] = np.sum(-tmp_product * sub_XXT, axis=0).ravel()
            hessian[-1, i * 2: (i + 1) * 2] = hessian[i * 2: (i + 1) * 2, -1]

        hessian[-1, -1] = np.sum(outputs[:, -1] * XXT[:, k - 1, k - 1] - outputs[:, -1] * outputs[:, -1] * XXT[:, k - 1, k - 1])

    elif method is 'FixDiag':

        hessian = np.zeros([k, k])

        for i in range(0, k - 1):
            for j in range(0, k - 1):
                if i <= j:
                    tmp_diff = outputs[:, i] * (int(i == j) - outputs[:, j])
                    hessian[i + 1, j + 1] = np.sum(tmp_diff * XXT[:, k - 1, k - 1], axis=0)
                else:
                    hessian[i + 1, j + 1] = hessian[j + 1, i + 1]

        tmp_product = np.sum(outputs[:, :-1] * X[:, :-1], axis=1).reshape(-1, 1).repeat(k - 1, 1) * outputs[:, :-1] * X[:, :-1]

        hessian[0, 0] = np.sum(np.sum(outputs[:, :-1] * (X[:, :-1] ** 2), axis=1) - np.sum(tmp_product, axis=1))

        tmp_product = np.sum(outputs[:, :-1] * X[:, :-1], axis=1).reshape(-1, 1).repeat(k - 1, 1) * outputs[:, :-1] * X[:, -1].reshape(-1, 1).repeat(k - 1, 1)

        for i in range(0, k - 1):
            hessian[0, i + 1] = np.sum(outputs[:, i] * X[:, i] * X[:, -1] - tmp_product[:, i])
            hessian[i + 1, 0] = hessian[0, i + 1]

    #np.set_printoptions(precision=1)
    #logging.debug('hessian is:')
    #if not (np.all(np.linalg.eigvals(hessian) > 0)):
    #    logging.debug('non-positive-definite Hessian is detected!')
    #logging.debug(hessian)
    np.nan_to_num(hessian, copy=False)
    return hessian


def _calculate_outputs(weights, X):
    k = np.shape(weights)[0]
    mul = np.dot(X, weights.transpose())
    return _softmax(mul)


def _softmax(X):
    """Compute the softmax of matrix X in a numerically stable way."""
    shiftx = X - np.max(X, axis=1).reshape(-1, 1)
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=1).reshape(-1, 1)
