#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

from svm import BinarySVM

import unittest

import sklearn.datasets
import numpy as np


def generate_guassian(mean, covar, samples):
    return np.random.multivariate_normal(mean, covar, samples)


def generate_real_separator():
    w, b = np.random.rand(2, 1), np.random.rand(1)
    w = w / np.linalg.norm(w)
    return (w, b)

def linearly_separable_dataset(w, b):
    return np.vstack(
        (
            generate_guassian([3, 0], [[0, 10], [0.1, 10]], 500),
            generate_guassian([-3, 0], [[0, 10], [0.1, 10]], 500)
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

    def test_synthetic(self):
        # w, b = generate_real_separator()
        X, y = linearly_separable_dataset(np.array([0, 1]), 0)
        model = BinarySVM(max_iter=1000, C=.01)
        model.fit(X, y)
        print(model.w/ np.linalg.norm(model.w), model.b, model.error)


def main():
    unittest.main()


if __name__ == '__main__':
    main()