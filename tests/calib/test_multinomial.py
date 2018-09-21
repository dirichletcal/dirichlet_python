import unittest
import numpy as np
from dirichlet.calib.multinomial import MultinomialRegression
from dirichlet.calib.multinomial import _get_weights
from . import get_simple_binary_example
from . import get_simple_ternary_example


class TestMultinomial(unittest.TestCase):
    def test_fit_predict(self):
        S, y = get_simple_binary_example()
        self.mlr = MultinomialRegression()
        self.mlr.fit(S, y)
        predictions = self.mlr.predict_proba(S).argmax(axis=1)
        self.assertTrue(np.alltrue(np.equal(predictions, y)))

        S, y = get_simple_ternary_example()
        self.mlr = MultinomialRegression()
        self.mlr.fit(S, y)
        predictions = self.mlr.predict_proba(S).argmax(axis=1)
        np.testing.assert_array_equal(predictions, y)


    def test_get_weights(self):
        k = 3
        params = np.arange((k-1)*(k+1)) + 1
        full_matrix = _get_weights(params, k=k, method='Full')
        expected = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 0]], 'float')
        np.testing.assert_array_equal(full_matrix, expected)

        k = 3
        params = np.arange(k + k - 1) + 1
        full_matrix = _get_weights(params, k=k, method='Diag')
        expected = np.array([[1, 0, 0, 2], [0, 3, 0, 4], [0, 0, 5, 0]], 'float')
        np.testing.assert_array_equal(full_matrix, expected)

        k = 3
        params = np.arange(k) + 1
        full_matrix = _get_weights(params, k=k, method='FixDiag')
        expected = np.array([[1, 0, 2], [0, 1, 3], [0, 0, 0]], 'float')
        np.testing.assert_array_equal(full_matrix, expected)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
