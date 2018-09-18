import unittest
import numpy as np
from dirichlet.calib.multinomial import MultinomialRegression


class TestMultinomial(unittest.TestCase):
    def setUp(self):
        self.mlr = MultinomialRegression()

    def test_fit(self):
        S = np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]).T
        S = np.hstack((np.flip(S, axis=0), S))
        y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.mlr.fit(S, y)
        predictions = self.mlr.predict_proba(S).argmax(axis=1)
        self.assertTrue(np.alltrue(np.equal(predictions, y)))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
