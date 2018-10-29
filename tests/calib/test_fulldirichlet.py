import unittest
import numpy as np
from dirichlet.calib.fulldirichlet import FullDirichletCalibrator
from . import get_simple_binary_example
from . import get_simple_ternary_example


class TestFullDirichlet(unittest.TestCase):
    def setUp(self):
        self.cal = FullDirichletCalibrator()

    def test_fit_predict(self):
        S, y = get_simple_binary_example()
        self.cal.fit(S, y)
        predictions = self.cal.predict_proba(S).argmax(axis=1)
        np.testing.assert_array_equal(predictions, y)

        S, y = get_simple_ternary_example()
        self.cal.fit(S, y)
        predictions = self.cal.predict_proba(S).argmax(axis=1)
        np.testing.assert_array_equal(predictions, y)

    def test_extreme_values(self):
        tiny = 10e-17 # np.finfo(float).tiny
        S = np.array([[tiny, tiny*10, 1.0-(tiny*10), 1.0-tiny],
                      [1.0-tiny, 1.0-(tiny*10), tiny*10, tiny]]).T
        y = np.array([0, 0, 1, 1])
        self.cal.fit(S, y)
        predictions = self.cal.predict_proba(S).argmax(axis=1)
        np.testing.assert_array_equal(predictions, y)

        tiny = 10e-18 # np.finfo(float).tiny
        S = np.array([[tiny, tiny*10, 1.0-(tiny*10), 1.0-tiny],
                      [1.0-tiny, 1.0-(tiny*10), tiny*10, tiny]]).T
        y = np.array([0, 0, 1, 1])
        self.cal.fit(S, y)
        predictions = self.cal.predict_proba(S).argmax(axis=1)
        np.testing.assert_array_equal(predictions, y)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
