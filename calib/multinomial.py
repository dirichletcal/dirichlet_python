from __future__ import division

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import label_binarize
from sklearn.metrics import log_loss

from scipy.optimize import minimize

import scipy
import scipy.optimize


def new_minimize_trust_region(fun, x0, args=(), jac=None, hess=None, hessp=None,
                           subproblem=None, initial_trust_radius=1.0,
                           max_trust_radius=1000.0, eta=0.15, gtol=1e-4,
                           maxiter=None, disp=False, return_all=False,
                           callback=None, inexact=True, change_ratio=0.25,
                           **unknown_options):
    """
    Minimization of scalar function of one or more variables using a
    trust-region algorithm.

    Options for the trust-region algorithm are:
        initial_trust_radius : float
            Initial trust radius.
        max_trust_radius : float
            Never propose steps that are longer than this value.
        eta : float
            Trust region related acceptance stringency for proposed steps.
        gtol : float
            Gradient norm must be less than `gtol`
            before successful termination.
        maxiter : int
            Maximum number of iterations to perform.
        disp : bool
            If True, print convergence message.
        inexact : bool
            Accuracy to solve subproblems. If True requires less nonlinear
            iterations, but more vector products. Only effective for method
            trust-krylov.

    This function is called by the `minimize` function.
    It is not supposed to be called directly.
    """
    print('new_minimize_trust_region')
    _check_unknown_options(unknown_options)

    if jac is None:
        raise ValueError('Jacobian is currently required for trust-region '
                         'methods')
    if hess is None and hessp is None:
        raise ValueError('Either the Hessian or the Hessian-vector product '
                         'is currently required for trust-region methods')
    if subproblem is None:
        raise ValueError('A subproblem solving strategy is required for '
                         'trust-region methods')
    if not (0 <= eta < 0.25):
        raise Exception('invalid acceptance stringency')
    if max_trust_radius <= 0:
        raise Exception('the max trust radius must be positive')
    if initial_trust_radius <= 0:
        raise ValueError('the initial trust radius must be positive')
    if initial_trust_radius >= max_trust_radius:
        raise ValueError('the initial trust radius must be less than the '
                         'max trust radius')

    # force the initial guess into a nice format
    x0 = np.asarray(x0).flatten()

    # Wrap the functions, for a couple reasons.
    # This tracks how many times they have been called
    # and it automatically passes the args.
    nfun, fun = wrap_function(fun, args)
    njac, jac = wrap_function(jac, args)
    nhess, hess = wrap_function(hess, args)
    nhessp, hessp = wrap_function(hessp, args)

    # limit the number of iterations
    if maxiter is None:
        maxiter = len(x0)*200

    # init the search status
    warnflag = 0

    # initialize the search
    trust_radius = initial_trust_radius
    x = x0
    if return_all:
        allvecs = [x]
    m = subproblem(x, fun, jac, hess, hessp)
    k = 0

    # search for the function min
    while True:

        # Solve the sub-problem.
        # This gives us the proposed step relative to the current position
        # and it tells us whether the proposed step
        # has reached the trust region boundary or not.
        try:
            p, hits_boundary = m.solve(trust_radius)
        except np.linalg.linalg.LinAlgError as e:
            warnflag = 3
            break

        # calculate the predicted value at the proposed point
        predicted_value = m(p)

        # define the local approximation at the proposed point
        x_proposed = x + p
        m_proposed = subproblem(x_proposed, fun, jac, hess, hessp)

        # evaluate the ratio defined in equation (4.4)
        actual_reduction = m.fun - m_proposed.fun
        predicted_reduction = m.fun - predicted_value
        if predicted_reduction <= 0:
            warnflag = 2
            break
        rho = actual_reduction / predicted_reduction

        # update the trust radius according to the actual/predicted ratio
        if rho < 0.25:
            trust_radius *= change_ratio
        elif rho > 0.75 and hits_boundary:
            trust_radius = min(2*trust_radius, max_trust_radius)

        # if the ratio is high enough then accept the proposed step
        if rho > eta:
            x = x_proposed
            m = m_proposed

        # append the best guess, call back, increment the iteration count
        if return_all:
            allvecs.append(np.copy(x))
        if callback is not None:
            callback(np.copy(x))
        k += 1

        # check if the gradient is small enough to stop
        if m.jac_mag < gtol:
            warnflag = 0
            break

        # check if we have looked at enough iterations
        if k >= maxiter:
            warnflag = 1
            break

    # print some stuff if requested
    status_messages = (
            _status_message['success'],
            _status_message['maxiter'],
            'A bad approximation caused failure to predict improvement.',
            'A linalg error occurred, such as a non-psd Hessian.',
            )
    if disp:
        if warnflag == 0:
            print(status_messages[warnflag])
        else:
            print('Warning: ' + status_messages[warnflag])
        print("         Current function value: %f" % m.fun)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % nfun[0])
        print("         Gradient evaluations: %d" % njac[0])
        print("         Hessian evaluations: %d" % nhess[0])

    result = OptimizeResult(x=x, success=(warnflag == 0), status=warnflag,
                            fun=m.fun, jac=m.jac, nfev=nfun[0], njev=njac[0],
                            nhev=nhess[0], nit=k,
                            message=status_messages[warnflag])

    if hess is not None:
        result['hess'] = m.hess

    if return_all:
        result['allvecs'] = allvecs

    return result


scipy.optimize._trustregion._minimize_trust_region.__code__ = new_minimize_trust_region.__code__


class MultinomialRegression(BaseEstimator, RegressorMixin):
    def __init__(self, weights_0=None, bounds=None):
        self.coef_ = None
        self.intercept_ = None
        self.weights_0_ = weights_0
        self.bounds_ = None

    def fit(self, X, y, *args, **kwargs):

        X_ = np.hstack((X, np.ones((len(X), 1))))

        classes = np.unique(y)
        k = len(classes)
        target = label_binarize(y, classes)
        if k == 2:
            target = np.hstack([target, 1-target])

        if self.weights_0_ is None:
            weights_0 = np.random.randn(k + 1, k - 1)
            weights_0[np.diag_indices(k - 1)] = np.abs(weights_0[np.diag_indices(k - 1)])
            weights_0[k - 1] = np.abs(weights_0[k - 1]) * -1
        else:
            weights_0 = self.weights_0_

        weights_0 = weights_0.transpose().ravel()

        # if self.bounds_ is None:
        #    dims = (k+1, k-1)
        #    diag_ravel_ind = np.ravel_multi_index(np.diag_indices(k-1), dims)

        #    k_ind = [np.ones(k-1, dtype=int)*(k-1), np.arange(k-1)]
        #    k_ravel_ind = np.ravel_multi_index(k_ind, dims)

        #    bounds = []
        #    for ind, _ in enumerate(weights_0):
        #        if ind in diag_ravel_ind:
        #            bounds.append((0, np.inf))
        #        elif ind in k_ravel_ind:
        #            bounds.append((-np.inf, 0))
        #        else:
        #            bounds.append((-np.inf, np.inf))
        # else:
        #    bounds = self.bounds_

        res = minimize(
            method='trust-exact',
            fun=_objective,
            jac=_gradient,
            hess=_hessian,
            x0=weights_0,
            args=(X_, target, k),
            bounds=None,
            #tol=1e-16,
            options={'disp': False,
                     'initial_trust_radius': 1.0,
                     'max_trust_radius': 1e32,
                     'change_ratio': 1 - 1e-3,
                     'eta': 0.0,
                     'maxiter': 1e4,
                     'gtol': 1e-8}
        )

        weights = res.x

        print('===================================================================')
        if res.success:
            print('optimisation converged!')
        else:
            print('optimisation not converged!')

        np.set_printoptions(precision=3)
        print('gradient is:')
        print(_gradient(weights, X_, target, k).reshape(-1, k+1).transpose())
        print('mean target is:')
        print(np.mean(target, axis=0))
        print('mean output is:')
        print(np.mean(_calculate_outputs(_get_weights(weights, k), X_), axis=0))
        print('reason for termination:')
        print(res.message)
        print('===================================================================')

        self.weights_ = _get_weights(weights, k)
        self.coef_ = self.weights_.transpose()[:, :-1]
        self.intercept_ = self.weights_.transpose()[:, -1]
        return self

    def predict_proba(self, S):
        S_ = np.hstack((S, np.ones((len(S), 1))))
        return _calculate_outputs(self.weights_, S_)

    def predict(self, S):
        return self.predict_proba(S)


def _get_weights(params, k):
    n_params = len(params)
    if n_params == k ** 2 - 1:
        return params.reshape(-1, k + 1).transpose()
    else:
        value = params[-1]
        intercepts = params[:-1]
        weights = np.zeros((k + 1, k - 1))
        weights[np.diag_indices(k - 1)] = value
        weights[k - 1] = value * -1
        weights[k] = intercepts
        return weights


def _objective(params, *args):
    (X, y, k) = args
    weights = _get_weights(params, k)
    outputs = _calculate_outputs(weights, X)
    loss = log_loss(y, outputs)
    #print('Loss is:')
    #print(loss)
    #print('Parameter is:')
    #print(weights)
    return loss


def _gradient(params, *args):
    (X, y, k) = args
    weights = _get_weights(params, k)
    outputs = _calculate_outputs(weights, X)
    graident = np.zeros((k + 1, k - 1))
    for i in range(0, k - 1):
        graident[:, i] = np.sum((outputs[:, i] - y[:, i]).reshape(-1, 1).repeat(k+1, axis=1) * X, axis=0)
    #print(graident)
    return graident.transpose().ravel()


def _hessian(params, *args):
    (X, y, k) = args
    weights = _get_weights(params, k)
    outputs = _calculate_outputs(weights, X)
    hessian = np.zeros((k**2 - 1, k**2 - 1))
    n = np.shape(X)[0]
    XXT = np.zeros((n, k+1, k+1))
    for i in range(0, n):
        XXT[i, :, :] = np.matmul(X[i, :].reshape(-1, 1), X[i, :].reshape(-1, 1).transpose())
    for i in range(0, k-1):
        for j in range(0, k-1):
            if i <= j:
                tmp_diff = outputs[:, i] * (int(i == j) - outputs[:, j])
                tmp_diff = tmp_diff.ravel().repeat((k+1)**2).reshape(n, k+1, k+1)
                hessian[i*(k+1):(i+1)*(k+1), j*(k+1):(j+1)*(k+1)] = np.sum(tmp_diff * XXT, axis=0)
            else:
                hessian[i*(k+1):(i+1)*(k+1), j*(k+1):(j+1)*(k+1)] = hessian[j*(k+1):(j+1)*(k+1), i*(k+1):(i+1)*(k+1)]

    #np.set_printoptions(precision=1)
    #print('hessian is:')
    #if not (np.all(np.linalg.eigvals(hessian) > 0)):
    #    print('non-positive-definite Hessian is detected!')
    #print(hessian)
    return hessian


def _calculate_outputs(weights, X):
    k = len(weights) - 1
    mul = np.zeros((len(X), k))
    mul[:, :k - 1] = np.dot(X, weights)
    return _softmax(mul)


def _softmax(X):
    """Compute the softmax of matrix X in a numerically stable way."""
    shiftx = X - np.max(X, axis=1).reshape(-1, 1)
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=1).reshape(-1, 1)
