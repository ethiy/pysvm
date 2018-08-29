#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

from svm import BinarySVM

import unittest

import sklearn.datasets
import numpy as np

import matplotlib.pyplot as plt

def generate_guassian(mean, covar, samples):
    return [np.random.multivariate_normal(mean, covar, 1) for s in range(samples)]


def synthetic_separable():
    return (
        generate_guassian([3, 0], [[0.1, 0.1], [0.1, 10]], 500)
        +
        generate_guassian([-3, 0], [[0.1, 0.1], [0.1, 10]], 500)
    ), (
        500 * [1]
        +
        500 * [-1]
    )


class BinaryTest(unittest.TestCase):
    """
    Test case for the 'BinarySVM' 'svm' class.
    """

    def test_synthetic_separable(self):
        X, y = synthetic_separable()
        model = BinarySVM(max_iter=100, C=100)
        model.fit(X, y)
        w = model.w()
        self.assertLessEqual(abs(w[1] / np.linalg.norm(w)), .01)

    
    def test_synthetic_cocentric(self):
        X, y = sklearn.datasets.make_circles(n_samples=1000, factor=.3, noise=.05)
        model = BinarySVM(max_iter=100, C=100, kernel=lambda x, y: x.dot(y.T)**2)
        model.fit([np.array(x) for x in X], (2*y-1).tolist())
        self.assertLess(abs(model.b + 1.8), .1)

def main():
    unittest.main()


if __name__ == '__main__':
    main()