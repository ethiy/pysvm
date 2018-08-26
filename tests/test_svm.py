#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

from svm import BinarySVM

import unittest

import sklearn.datasets
import numpy as np

import matplotlib.pyplot as plt

def generate_guassian(mean, covar, samples):
    return np.random.multivariate_normal(mean, covar, samples)


def synthetic_separable():
    return np.vstack(
        (
            generate_guassian([3, 0], [[0.1, 0.1], [0.1, 10]], 500),
            generate_guassian([-3, 0], [[0.1, 0.1], [0.1, 10]], 500)
        )
    ), np.concatenate(
        (
            np.ones(500),
            - np.ones(500)
        )
    )


class BinaryTest(unittest.TestCase):
    """
    Test case for the 'BinarySVM' 'svm' class.
    """

    def test_synthetic_separable(self):
        X, y = synthetic_separable()
        model = BinarySVM(max_iter=1000, epsilon=1E-15, C=10)
        model.fit(X, y)
        self.assertLessEqual(abs(model.w[1] / np.linalg.norm(model.w)), .05)

    
    def test_synthetic_cocentric(self):
        X, y = sklearn.datasets.make_circles(n_samples=1000, factor=.3, noise=.05)
        model = BinarySVM(max_iter=300, epsilon=1E-15, C=2, kernel=lambda x, y: x.dot(y)**4)
        model.fit(X, 2*y-1)
        self.assertLessEqual(abs(model.b), .01)

def main():
    unittest.main()


if __name__ == '__main__':
    main()