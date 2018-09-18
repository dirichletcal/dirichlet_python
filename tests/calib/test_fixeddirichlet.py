import unittest
import numpy as np
from dirichlet.calib.fixeddirichlet import FixedDiagonalDirichletCalibrator


class TestFixedDiagonalDirichlet(unittest.TestCase):
    def setUp(self):
        self.cal = FixedDiagonalDirichletCalibrator()

    def test_fit(self):
        S = np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]).T
        S = np.hstack((np.flip(S, axis=0), S))
        y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.cal.fit(S, y)
        predictions = self.cal.predict_proba(S).argmax(axis=1)
        np.testing.assert_array_equal(predictions, y)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
