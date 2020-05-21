from __future__ import absolute_import, division, print_function

import tensorflow as tf

import tensorflow.contrib.eager as tfe

import numpy

from sklearn.base import BaseEstimator, RegressorMixin

tf.enable_eager_execution()


class TypeIIDirichletCalibrator(BaseEstimator, RegressorMixin):

    def __init__(self):

        self.k = None

        self.theta_w = None

    def fit(self, x, y, lr=0.05, batch_size=1024,
            sample_size=128, ftol=1e-4, max_batch=int(1e8), **kwargs):

        n_y = numpy.shape(x)[0]

        k = numpy.shape(x)[1]

        self.k = k

        feature_x = numpy.hstack([numpy.log(x), numpy.ones((n_y, 1))])

        feature_x[numpy.isinf(feature_x)] = 0.0

        y_hot = numpy.zeros((n_y, k))

        theta_w = numpy.zeros((2*k, k+1))

        theta_w[k:, :] = 1.0

        for i in range(0, k):
            y_hot[y == i, i] = 1.0
            theta_w[i, i] = 1.0

        fin_theta_w = theta_w.copy()

        theta_w = tf.Variable(theta_w, 'Theta')

        get_obj_g = tfe.gradients_function(self._objective, params=[0])

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        D = tf.data.Dataset.from_tensor_slices((tf.cast(feature_x, tf.float64), tf.cast(y_hot, tf.float64)))

        D = D.shuffle(1024).batch(batch_size)

        batch_L = []

        mean_L = []

        for (batch, (f_x, y_h)) in enumerate(D.repeat(1e8)):

            L_t = self._objective(theta_w, f_x, y_h, k, sample_size)

            batch_L.append(L_t)

            g_t = tf.reduce_mean(get_obj_g(theta_w, f_x, y_h, k, sample_size), axis=0)

            optimizer.apply_gradients([(g_t, theta_w)],
                                      global_step=tf.train.get_or_create_global_step())

            if len(batch_L) >= 8:
                mean_L.append(numpy.mean(batch_L[-8:]))
            else:
                for o in range(0, len(mean_L)):
                    mean_L[o] = numpy.mean(batch_L)
                mean_L.append(numpy.mean(batch_L))

            if len(mean_L) >= 2:
                if mean_L[-1] < numpy.min(mean_L[:-1]):
                    fin_theta_w = theta_w.numpy().copy()

            if len(mean_L) > 128:

                previous_opt = numpy.min(mean_L.copy()[:-128])

                current_opt = numpy.min(mean_L.copy()[-128:])

                if previous_opt - current_opt <= ftol:
                    print(previous_opt, current_opt)
                    break

                if len(mean_L) >= max_batch:
                    break

        self.theta_w = fin_theta_w

        return self

    def _objective(self, theta_w, feature_x, y_hot, k, sample_size):

        mu_w = tf.reshape(tf.transpose(theta_w[:k, :]), [1, -1])

        sigma_w = tf.reshape(tf.transpose(theta_w[k:, :]), [1, -1])

        e = tf.random.normal((sample_size, k*(k+1)), dtype=tf.float64)

        sample_w = tf.reshape((e * ((sigma_w**2)**0.5) + mu_w), [sample_size, k+1, k])

        raw_prod = tf.tensordot(feature_x, sample_w, axes=[[1], [1]])

        prod = raw_prod - tf.expand_dims(tf.reduce_max(raw_prod, axis=2), axis=2)

        p_y = tf.reduce_mean(tf.exp(prod) / tf.expand_dims(tf.reduce_sum(tf.exp(prod), axis=2), 2),
                             axis=1)

        brier = tf.reduce_mean(tf.reduce_sum((p_y - y_hot)**2, axis=1))

        return brier

    def predict_proba(self, x, sample_size=1024):

        n_y = numpy.shape(x)[0]

        feature_x = numpy.hstack([numpy.log(x), numpy.ones((n_y, 1))])

        feature_x[numpy.isinf(feature_x)] = 0.0

        k = numpy.shape(x)[1]

        mu_w = self.theta_w[:k, :].transpose().ravel()

        sigma_w = self.theta_w[k:, :].transpose().ravel()

        e = tf.random.normal((sample_size, k * (k + 1)), dtype=tf.float64)

        sample_w = tf.reshape((e * ((sigma_w ** 2) ** 0.5) + mu_w), [sample_size, k + 1, k])

        raw_prod = tf.tensordot(feature_x, sample_w, axes=[[1], [1]])

        prod = raw_prod - tf.expand_dims(tf.reduce_max(raw_prod, axis=2), axis=2)

        p_y = tf.reduce_mean(tf.exp(prod) / tf.expand_dims(tf.reduce_sum(tf.exp(prod), axis=2), 2),
                             axis=1)

        return p_y.numpy()

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

