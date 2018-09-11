from __future__ import division

import logging
logger = logging.getLogger(__name__)


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
    import logging
    logger = logging.getLogger(__name__)
    logger.debug('new_minimize_trust_region')
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
            if np.isnan(trust_radius).any():
                logger.debug(trust_radius)
            # FIXME Ocasionally the inner computation overflows
            # OverflowError: (34, 'Numerical result out of range')
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
        try:
            actual_reduction = m.fun - m_proposed.fun
        except ValueError as e:
            print(e)
            raise e
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
            logger.debug(status_messages[warnflag])
        else:
            logger.debug('Warning: ' + status_messages[warnflag])
        logger.debug("         Current function value: %f" % m.fun)
        logger.debug("         Iterations: %d" % k)
        logger.debug("         Function evaluations: %d" % nfun[0])
        logger.debug("         Gradient evaluations: %d" % njac[0])
        logger.debug("         Hessian evaluations: %d" % nhess[0])

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
    def __init__(self, weights_0=None, method='Full'):
        # Method in {'Full', 'Diag', 'FixDiag'}
        self.coef_ = None
        self.intercept_ = None
        self.weights_0_ = weights_0
        self.method_ = method

    def fit(self, X, y, *args, **kwargs):
        logger = logging.getLogger(__name__)

        X_ = np.hstack((X, np.ones((len(X), 1))))

        classes = np.unique(y)

        k = len(classes)

        target = label_binarize(y, classes)

        if k == 2:
            target = np.hstack([target, 1-target])

        if self.weights_0_ is None:

            if self.method_ is 'Full':

                weights_0 = np.random.randn((k - 1) * (k + 1))

                weights_0 = _get_weights(weights_0, k, self.method_)

                weights_0[np.diag_indices(k - 1)] = np.abs(weights_0[np.diag_indices(k - 1)])

                weights_0[:, k - 1] = np.abs(weights_0[:, k - 1]) * -1

            elif self.method_ is 'Diag':

                weights_0 = np.random.randn(k + (k - 1))

                weights_0 = _get_weights(weights_0, k, self.method_)

                weights_0[np.diag_indices(k)] = np.abs(weights_0[np.diag_indices(k)])

            elif self.method_ is 'FixDiag':

                weights_0 = np.random.randn(k)

                weights_0[0] = np.abs(weights_0[0])

        else:
            weights_0 = self.weights_0_

        weights_0 = weights_0[weights_0!=0]

        n = np.shape(X_)[0]

        m = np.shape(X_)[1]

        XXT = np.zeros([n, m, m])

        for i in range(0, n):
            XXT[i, :, :] = np.matmul(X_[i, :].reshape(-1, 1), X_[i, :].reshape(-1, 1).transpose())


        logger.debug(self.method_)

        # res = minimize(
        #         method='trust-krylov',
        #         fun=_objective,
        #         jac=_gradient,
        #         hess=_hessian,
        #         x0=weights_0,
        #         args=(X_, XXT, target, k, self.method_),
        #         bounds=None,
        #         #tol=1e-16,
        #         options={'disp': False,
        #             'initial_trust_radius': 1.0,
        #             'max_trust_radius': 1e32,
        #             'change_ratio': 1 - 1e-4,
        #             'eta': 0.0,
        #             'maxiter': 5e4,
        #             'gtol': 1e-8}
        #         )

        #res = minimize(
        #    method='BFGS',
        #    fun=_objective,
        #    jac=_gradient,
        #    x0=weights_0,
        #    args=(X_, XXT, target, k, self.method_),
        #)

        # weights = res.x

        weights_0 = np.zeros_like(weights_0)

        weights = _newton_update(weights_0, X_, XXT, target, k, self.method_)

        # logger.debug('===================================================================')
        # if res.success:
        #     logger.debug('optimisation converged!')
        # else:
        #     logger.debug('optimisation not converged!')

        #np.set_printoptions(precision=3)
        #logger.debug('gradient is:')
        #logger.debug(_gradient(weights, X_, XXT, target, k, self.method_))
        #logger.debug('mean target is:')
        #logger.debug(np.mean(target, axis=0))
        #logger.debug('mean output is:')
        #logger.debug(np.mean(_calculate_outputs(_get_weights(weights, k, self.method_), X_), axis=0))
        #logger.debug('obtained paprameters are:')
        #logger.debug(_get_weights(weights, k, self.method_))
        # logger.debug('reason for termination:')
        # logger.debug(res.message)
        #logger.debug('===================================================================')

        self.weights_ = _get_weights(weights, k, self.method_)
        self.coef_ = self.weights_[:, :-1]
        self.intercept_ = self.weights_[:, -1]
        return self

    def predict_proba(self, S):
        S_ = np.hstack((S, np.ones((len(S), 1))))
        return _calculate_outputs(self.weights_, S_)

    def predict(self, S):
        return self.predict_proba(S)
    

def _newton_update(weights_0, X, XX_T, target, k, method_, maxiter=int(1e3),
                   ftol=1e-12, gtol=1e-12):

    L_list = [_objective(weights_0, X, XX_T, target, k, method_)]

    weights = weights_0.copy()

    for i in range(0, maxiter):

        gradient = _gradient(weights, X, XX_T, target, k, method_)

        if np.abs(gradient).sum() < gtol:
            break

        hessian = _hessian(weights, X, XX_T, target, k, method_)

        for step_size in np.hstack((np.linspace(1, 0.1, 10),
                                    np.linspace(0.09, 0.01, 9),
                                    np.linspace(0.009, 0.001, 9),
                                    np.linspace(0.0009, 0.0001, 9),
                                    np.linspace(0.00009, 0.00001, 9),
                                    np.linspace(0.000009, 0.000001, 9),
                                    1e-8, 1e-16, 1e-32)):

            updates = (np.matmul(scipy.linalg.pinv2(hessian), gradient.reshape(-1, 1)) * step_size).ravel()

            tmp_w = weights - updates

            L = _objective(tmp_w, X, XX_T, target, k, method_)

            if (L - L_list[-1]) < 0:
                break

        L_list.append(L)

        logger.debug({'iteration': i, 'loss': L, 'sum(abs(grad))':
                      np.abs(gradient).sum()})

        if i >= 5:
            if (np.min(np.diff(L_list[-5:])) > -ftol) & (np.sum(np.diff(L_list[-5:]) > 0) == 0):
                weights = tmp_w.copy()
                logger.debug('terminate as there is not enough changes on Psi.')
                break

        if np.sum(np.diff(L_list[-2:]) > 0) == 1:
            logger.debug('terminate as the loss increased.')
            break
        else:
            weights = tmp_w.copy()

    logger.debug('Current Gradients Are:')
    logger.debug(gradient)

    return weights


def _get_weights(params, k, method):

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


def _objective(params, *args):
    (X, _, y, k, method) = args
    weights = _get_weights(params, k, method)
    outputs = _calculate_outputs(weights, X)
    eps = np.finfo(outputs.dtype).eps
    outputs = np.clip(outputs, eps, 1-eps)
    loss = log_loss(y, outputs)
    #logger.debug('Loss is:')
    #logger.debug(loss)
    #logger.debug('Parameter is:')
    #logger.debug(weights)
    return loss


def _gradient(params, *args):
    import logging
    logger = logging.getLogger(__name__)
    (X, _, y, k, method) = args
    weights = _get_weights(params, k, method)
    outputs = _calculate_outputs(weights, X)

    if method is 'Full':

        gradient = np.random.randn((k - 1) * (k + 1))

        for i in range(0, k - 1):

            gradient[i * (k + 1):(i + 1) * (k + 1)] = \
                    np.sum((outputs[:, i] - y[:, i]).reshape(-1, 1).repeat(k+1, axis=1) * X, axis=0)

    elif method is 'Diag':

        gradient = np.random.randn(k + (k - 1))

        for i in range(0, k - 1):

            gradient[i * 2:(i + 1) * 2] = \
                    np.sum((outputs[:, i] - y[:, i]).reshape(-1, 1).repeat(2, axis=1) * X[:, [i, k]], axis=0)

        gradient[-1] = np.sum((outputs[:, i] - y[:, i]) * X[:, k])

    elif method is 'FixDiag':

        gradient = np.random.randn(k)

        gradient[0] = np.sum((outputs[:, :-1] - y[:, :-1]) * X[:, :-1])

        for i in range(0, k - 1):

            gradient[i + 1] = np.sum((outputs[:, i] - y[:, i]) * X[:, k - 1])

    #logger.debug(gradient)
    np.nan_to_num(gradient, copy=False)
    return gradient


def _hessian(params, *args):
    import logging
    logger = logging.getLogger(__name__)
    (X, XXT, y, k, method) = args
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
                    hessian[i * (k + 1): (i + 1) * (k + 1), j * (k + 1): (j + 1) * (k + 1)] = np.sum(tmp_diff * XXT, axis=0)
                else:
                    hessian[i * (k + 1): (i + 1) * (k + 1), j * (k + 1): (j + 1) * (k + 1)] = \
                            hessian[j * (k + 1): (j + 1) * (k + 1), i * (k + 1): (i + 1) * (k + 1)]

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
    #logger.debug('hessian is:')
    #if not (np.all(np.linalg.eigvals(hessian) > 0)):
    #    logger.debug('non-positive-definite Hessian is detected!')
    #logger.debug(hessian)
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
