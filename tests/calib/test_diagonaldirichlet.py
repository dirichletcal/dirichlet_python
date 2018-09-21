import unittest
import numpy as np
from dirichlet.calib.diagdirichlet import DiagonalDirichletCalibrator
from . import get_simple_binary_example


class TestDiagonalDirichlet(unittest.TestCase):
    def setUp(self):
        self.cal = DiagonalDirichletCalibrator()

    def test_fit_predict(self):
        S, y = get_simple_binary_example()
        self.cal.fit(S, y)
        predictions = self.cal.predict_proba(S).argmax(axis=1)
        np.testing.assert_array_equal(predictions, y)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
