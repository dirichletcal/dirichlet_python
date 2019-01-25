import autograd
import autograd.numpy as numpy

from sklearn.base import BaseEstimator, RegressorMixin


class TypeIIDirichletCalibrator(BaseEstimator, RegressorMixin):

    def __init__(self):

        self.k = None

        self.theta_w = None

    def fit(self, x, y, batch_size=128, sample_size=128, lr=1e-2, beta_1=0.9, beta_2=0.999,
            eps=1e-8, max_batch=int(1e8), factr=1e-3):

        n_y = numpy.shape(x)[0]

        k = numpy.shape(x)[1]

        self.k = k

        feature_x = numpy.hstack([numpy.log(x), numpy.ones((n_y, 1))])

        y_hot = numpy.zeros((n_y, k))

        theta_w = numpy.zeros((2*k, k+1))

        for i in range(0, k):
            y_hot[y == i, i] = 1.0
            theta_w[i, i] = 1.0

        m = numpy.zeros_like(theta_w)

        v = numpy.zeros_like(theta_w)

        fin_theta_w = theta_w

        get_gradient = autograd.elementwise_grad(self._objective, 0)

        batch_idx = numpy.arange(0, n_y, batch_size)

        batch_num = len(batch_idx) - 1

        converge = False

        batch_L = []

        mean_L = []

        for i in range(0, int(1e8)):

            for j in range(0, batch_num):

                L_t = self._objective(theta_w, feature_x, y_hot, k, sample_size)

                g_t = get_gradient(theta_w, feature_x, y_hot, k, sample_size)

                m = beta_1 * m + (1 - beta_1) * g_t

                v = beta_2 * v + (1 - beta_2) * g_t * g_t

                theta_w = theta_w - lr * m / (v ** 0.5 + eps)

                batch_L.append(L_t)

                if len(batch_L) >= 8:
                    mean_L.append(numpy.mean(batch_L[-8:]))
                else:
                    for o in range(0, len(mean_L)):
                        mean_L[o] = numpy.mean(batch_L)
                    mean_L.append(numpy.mean(batch_L))

                if len(mean_L) >= 2:
                    if mean_L[-1] < numpy.min(mean_L[:-1]):
                        fin_theta_w = theta_w.copy()

                if len(mean_L) > 64:

                    previous_opt = numpy.min(mean_L.copy()[:-64])

                    current_opt = numpy.min(mean_L.copy()[-64:])

                    if previous_opt - current_opt <= numpy.abs(previous_opt * factr):

                        converge = True

                        break

                    if len(mean_L) >= max_batch:

                        converge = True

                        break

            per_idx = numpy.random.permutation(n_y)

            feature_x = feature_x[per_idx]

            y_hot = y_hot[per_idx]

            if converge:

                break

        self.theta_w = fin_theta_w

        return self

    def _objective(self, theta_w, feature_x, y_hot, k, sample_size):

        mu_w = theta_w[:k, :].transpose().ravel()

        sigma_w = theta_w[k:, :].transpose().ravel()

        e = numpy.random.randn(sample_size, k*(k+1))

        sample_w = (e * (sigma_w**2**0.5) + mu_w).reshape(sample_size, k+1, k)

        prod = numpy.matmul(feature_x, sample_w)

        p_y = numpy.mean(numpy.exp(prod) /
                         numpy.sum(numpy.exp(prod), axis=2)[:, :, numpy.newaxis].repeat(k, axis=2),
                         axis=0)

        brier = numpy.mean(numpy.sum((p_y - y_hot)**2, axis=1))

        return brier

    def predict_proba(self, x, sample_size=1024):

        feature_x = numpy.log(x)

        k = numpy.shape(x)[1]

        mu_w = self.theta_w[:k, :].transpose().ravel()

        sigma_w = self.theta_w[k:, :].transpose().ravel()

        e = numpy.random.randn(sample_size, k*(k+1))

        sample_w = (e * (sigma_w**2**0.5) + mu_w).reshape(sample_size, k+1, k)

        prod = numpy.matmul(feature_x, sample_w)

        p_y = numpy.mean(numpy.exp(prod) /
                         numpy.sum(numpy.exp(prod), axis=2)[:, :, numpy.newaxis].repeat(k, axis=2),
                         axis=0)

        return p_y

    def predict(self, x):

        return self.predict_proba(x)

    @property
    def coef_(self):
        return self.theta_w[:self.k, :self.k]

    @property
    def intercept_(self):
        return self.theta_w[:self.k, self.k]


if __name__ == '__main__':
    from sklearn import datasets

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :3]  # we only take the first two features.
    y = iris.target

    softmax = lambda z:numpy.divide(numpy.exp(z).T, numpy.sum(numpy.exp(z), axis=1)).T
    S = softmax(X)

    print(S)
    print(y)
    calibrator = TypeIIDirichletCalibrator()
    calibrator.fit(S, y)
    print(calibrator.predict_proba(S))










