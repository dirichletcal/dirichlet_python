import unittest
import numpy as np
from dirichletcal.calib.diagdirichlet import DiagonalDirichletCalibrator
from . import get_simple_binary_example
from . import get_simple_ternary_example

from sklearn.metrics import accuracy_score


class TestDiagonalDirichlet(unittest.TestCase):
    def setUp(self):
        # Removed while problems are fixed
        pass
        self.cal = DiagonalDirichletCalibrator()

    def test_fit_predict(self):
        for S, y in (get_simple_binary_example(),
                     get_simple_ternary_example()):
            self.cal.fit(S, y)
            predictions = self.cal.predict_proba(S).argmax(axis=1)
            acc = accuracy_score(y, predictions)
            self.assertGreater(acc, 0.97,
                               "accuracy must be superior to 97 percent")

            ac = self.cal.cannonical_weights
            self.assertAlmostEqual(sum(ac[:,-1]), 1)
            assert(np.all(ac >= 0))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
