from __future__ import division

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.metrics import log_loss

from scipy.optimize import fmin_l_bfgs_b


class _DiagonalDirichletCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y, *args, **kwargs):
        eps = np.finfo(X.dtype).eps
        X_ = np.log(np.clip(X, eps, 1-eps))

        X_ = np.hstack((X_, np.ones((len(X), 1))))

        k = len(np.unique(y))
        target = label_binarize(y, range(k))

        weights_0 = np.zeros((k+1, k-1))
        weights_0[np.diag_indices(k-1)] = np.random.rand(k-1)
        weights_0[k-1] = np.random.rand(k-1) * -1
        weights_0[k] = np.random.randn(k-1)        

        weights_0 = weights_0.ravel()

        dims = (k+1, k-1)
        diag_ravel_ind = np.ravel_multi_index(np.diag_indices(k-1), dims)

        k_ind = [np.ones(k-1, dtype=int)*(k-1), np.arange(k-1)]
        k_ravel_ind = np.ravel_multi_index(k_ind, dims)

        intercept_ind = [np.ones(k-1, dtype=int)*k, np.arange(k-1)]
        intercept_ravel_ind = np.ravel_multi_index(intercept_ind, dims)

        bounds = []
        for ind, _ in enumerate(weights_0):
            if ind in diag_ravel_ind:
                bounds.append((0, np.inf))
            elif ind in k_ravel_ind:
                bounds.append((-np.inf, 0))
            elif ind in intercept_ravel_ind:
                bounds.append((-np.inf, np.inf))
            else:
                bounds.append((0, 0))

        is_single = False

        weights, _, _ = fmin_l_bfgs_b(
            _objective,
            fprime=_grad,
            x0=weights_0,
            args=(X_, target, is_single),
            bounds=bounds,
            iprint=5
        )

        self.weights_ = weights.reshape(-1, k-1)
        self.coef_ = self.weights_.transpose()[:, :-1]
        self.intercept_ = self.weights_.transpose()[:, -1]
        return self

    def predict_proba(self, S):
        eps = np.finfo(S.dtype).eps
        S_ = np.log(np.clip(S, eps, 1-eps))
        S_ = np.hstack((S_, np.ones((len(S), 1))))
        return _calculate_outputs(self.weights_, S_)

    def predict(self, S):
        return self.predict_proba(S)


class _FixedDiagonalDirichletCalibrator(_DiagonalDirichletCalibrator):
    def fit(self, X, y, *args, **kwargs):
        eps = np.finfo(X.dtype).eps
        X_ = np.log(np.clip(X, eps, 1-eps))

        X_ = np.hstack((X_, np.ones((len(X), 1))))

        k = len(np.unique(y))
        target = label_binarize(y, range(k))

        weights_0 = np.append(np.random.randn(k-1), np.random.rand()*10)
        bounds = [(-np.inf, np.inf) for _ in range(k-1)]
        bounds.append((0, np.inf))

        is_single = True

        weights, _, _ = fmin_l_bfgs_b(
            _objective,
            fprime=_grad,
            x0=weights_0,
            args=(X_, target, is_single),
            bounds=bounds
        )

        self.weights_ = _get_weights(weights, is_single)
        self.coef_ = self.weights_.transpose()[:, :-1]
        self.intercept_ = self.weights_.transpose()[:, -1]
        return self


def _get_weights(params, is_single):
    if not is_single:
        n_params = len(params)
        k = int(np.ceil(np.sqrt(n_params - 1)))
        return params.reshape(-1, k-1)
    else:
        k = len(params)
        value = params[-1]
        intercepts = params[:-1]
        weights = np.zeros((k+1, k-1))
        weights[np.diag_indices(k-1)] = value
        weights[k-1] = value * -1
        weights[k] = intercepts
        return weights


def _objective(params, *args):
    (X, y, is_single) = args
    weights = _get_weights(params, is_single)
    outputs = _calculate_outputs(weights, X)
    loss = log_loss(y, outputs)
    return loss


def _grad(params, *args):
    (X, y, is_single) = args
    weights = _get_weights(params, is_single)
    outputs = _calculate_outputs(weights, X)

    k = len(weights) - 1

    grad = np.zeros_like(weights)

    s = outputs[range(len(y)), np.argmax(y, axis=1)]

    for i in range(k + 1):
        for j in range(k - 1):
            grad[i, j] = np.mean((s - 1) * X[:, i])

    if not is_single:
        grad = grad.ravel()
    else:
        grad_fixed = np.sum(grad[np.diag_indices(k-1)])
        grad_fixed += grad[k-1, 0]
        grad = np.append(grad[k], grad_fixed)
    return grad


def _calculate_outputs(weights, X):
    k = len(weights) - 1
    mul = np.zeros((len(X), k))
    mul[:, :k-1] = np.dot(X, weights)
    return _softmax(mul)


def _softmax(X):
    """Compute the softmax of matrix X in a numerically stable way."""
    shiftx = X - np.max(X, axis=1).reshape(-1, 1)
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=1).reshape(-1, 1)


class _FullDirichletCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.calibrator_ = None

    def fit(self, X, y, *args, **kwargs):
        n = len(y)

        eps = np.finfo(X.dtype).eps
        X = np.log(np.clip(X, eps, 1-eps))

        self.calibrator_ = LogisticRegression(
            C=99999999999,
            multi_class='multinomial', solver='saga'
        ).fit(X, y, *args, **kwargs)
        self.coef_ = self.calibrator_.coef_
        self.intercept_ = self.calibrator_.intercept_

        return self

    def predict_proba(self, S):
        eps = np.finfo(S.dtype).eps
        S = np.log(np.clip(S, eps, 1-eps))
        return self.calibrator_.predict_proba(S)

    def predict(self, S):
        eps = np.finfo(S.dtype).eps
        S = np.log(np.clip(S, eps, 1-eps))
        return self.calibrator_.predict(S)


class DirichletCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self, matrix_type='full'):
        if matrix_type == 'diagonal':
            self.calibrator_ = _DiagonalDirichletCalibrator()
        elif matrix_type == 'fixed_diagonal':
            self.calibrator_ = _FixedDiagonalDirichletCalibrator()
        else:
            self.calibrator_ = _FullDirichletCalibrator()

    def fit(self, X, y, *args, **kwargs):
        self.calibrator_ = self.calibrator_.fit(X, y, *args, **kwargs)
        self.coef_ = self.calibrator_.coef_
        self.intercept_ = self.calibrator_.intercept_
        return self

    def predict_proba(self, S):
        return self.calibrator_.predict_proba(S)

    def predict(self, S):
        return self.calibrator_.predict(S)

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.naive_bayes import GaussianNB

    iris = load_iris()
    nb = GaussianNB().fit(iris.data, iris.target)
    pred = nb.predict_proba(iris.data)

    diag = DirichletCalibrator(matrix_type='diagonal').fit(pred, iris.target)
    print diag.coef_
    print diag.intercept_

    print 'll diag: {}'.format(log_loss(iris.target, diag.predict_proba(pred)))

    # fixed = DirichletCalibrator(
    #     matrix_type='fixed_diagonal').fit(pred, iris.target)
    # print fixed.coef_
    # print fixed.intercept_

    # print 'll fixed: {}'.format(
    #     log_loss(iris.target, fixed.predict_proba(pred)))

    # full = DirichletCalibrator(matrix_type='full').fit(pred, iris.target)
    # print full.coef_
    # print full.intercept_

    # print 'll full: {}'.format(log_loss(iris.target, full.predict_proba(pred)))