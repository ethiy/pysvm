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

    # def test_synthetic_separable(self):
    #     X, y = synthetic_separable()
    #     model = BinarySVM(max_iter=20, C=1, debug=True, verbose=True)
    #     model.fit(X, y)
    #     plt.plot(model.lds)
    #     plt.show()
    #     w = model.w()
    #     print(w/np.linalg.norm(w))
    #     print(np.argwhere(np.isnan(model.K)).shape)
    #     print(model.b_up,model.b_low)
        # self.assertLessEqual(abs(w[1] / np.linalg.norm(w)), .01)

    
    def test_synthetic_cocentric(self):
        X, y = sklearn.datasets.make_circles(n_samples=1000, factor=.3, noise=.05)
        model = BinarySVM(max_iter=100, C=10, debug=True, verbose=True, kernel=lambda x, y: x.dot(y.T)**2)
        model.fit([np.array(x) for x in X], (2*y-1).tolist())
        plt.plot(model.lds)
        plt.show()
        print(np.argwhere(np.isnan(model.K)).shape)
        print(model.b_up,model.b_low)
        # print([model.phi(model.X[s]) - model.Y[s] for s in model.support_vectors_idx])
    #     # self.assertLessEqual(abs(model.b), .000001)

def main():
    unittest.main()


if __name__ == '__main__':
    main()