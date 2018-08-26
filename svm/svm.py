import math
import numpy as np
import random

import sklearn.base
import sklearn.multiclass


def random_int_except(m, M, ex):
    return random.choice(
        [n for n in range(m, M) if n not in ex]
    )


class BinarySVM(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """
        SVM implemented using SMO.
        Parameters
        ----------
        verbose : bool, optional
            Define if messages will be printed on stdout.
        Attributes
        ----------
        alpha : 
    """

    def __init__(self, max_iter=math.inf, kernel=lambda x, y: x.T.dot(y), C=1.0, epsilon=0, debug=False):
        self.max_iter = max_iter
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.debug = debug

        self.n, self.d = (0, 0)
        self.initialize()

        self.error = math.inf
        self.errors = [self.error] if self.debug else None
        self.support_vectors = np.zeros((0, self.d))

    def initialize(self):
        self.K = np.full((self.n, self.n), np.nan)
        self.eta = dict()
        self.alpha = np.zeros(self.n)
        self.w = np.zeros(self.d)
        self.b = 0
    
    def train_iterations(self):
        iteration = 0
        while iteration < self.max_iter and self.error > self.epsilon:
            iteration += 1
            yield iteration

    def update_separator(self, X, y, sample_weight=None):
        self.w = np.dot(self.alpha * y, X)
        self.b = np.mean(y - np.dot(self.w.T, X.T))

    def score(self, X, y, sample_weight=None):
        pass

    def update_a_j(self, i, j, X, y):
        L, H = self.bounds(i, j, y)
        a_j_old = self.alpha[j]
        dd = self.decision_discrepancy(X, y, i, j)
        if self.eta[(min(i, j), max(i, j))] == 0:
            self.alpha[j] = L if dd * L < dd * H else H
        else:
            self.alpha[j] += self.delta(dd, i, j)
            self.clip(j, (L, H))
        return self.alpha[j] - a_j_old

    def update_couple(self, i, j, X, y):
        self.compute_eta(i, j, X, y)
        self.alpha[i] += - y[j] * y[i] * self.update_a_j(i, j, X, y)
        self.update_separator(X, y)

    def fit(self, X, y, sample_weight=None):
        self.n, self.d = X.shape
        self.initialize()
        self.compute_diagonal(X)
        for iteration in self.train_iterations():
            print(iteration, self.error)
            prev_alpha = np.copy(self.alpha)
            for i in range(self.n):
                self.update_couple(
                    i,
                    random_int_except(0, self.n, [i]),
                    X,
                    y
                )
            self.error = np.linalg.norm(self.alpha - prev_alpha)
            if self.debug:
                self.errors.append(self.error)
        self.compute_support(X)
        return self
    
    def compute_diagonal(self, X):
        for i in range(self.n):
            self.K[i, i] = self.kernel(X[i, :], X[i, :])

    def compute_support(self, X):
        self.support_vectors = np.vstack(
            [
                X[idx, :]
                for (idx, a) in enumerate(self.alpha)
                if a > 0
            ]
        )
    
    def distance(self, X):
        return np.dot(self.w.T, X.T) + self.b

    def predict(self, X):
        return np.sign(self.distance(X)).astype(int) 
    
    def E(self, x, y):
        return self.distance(x) - y
    
    def compute_eta(self, i, j, X, y):
        if (min(i, j), max(i, j)) not in self.eta.keys():
            if np.isnan(self.K[i,j]):
                self.K[i, j] = self.kernel(X[i, :], X[j, :])
                self.K[j, i] = self.K[i, j]
            self.eta[(min(i, j), max(i, j))] = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
    
    def decision_discrepancy(self, X, y, i, j):
        return float(y[j] * (self.E(X[j, :], y[j]) - self.E(X[i, :], y[i])))

    def delta(self, dd, i, j):
        return dd / self.eta[(min(i, j), max(i, j))]
    
    def bounds(self, i, j, y):
        return (
            max(0, self.alpha[j] - (y[i] == y[j]) * self.C + y[i] * y[j] * self.alpha[i]),
            min(self.C, self.alpha[j] + (y[i] != y[j]) * self.C + y[i] * y[j] * self.alpha[i])
        )

    def clip(self, i, bounds):
        self.alpha[i] = max(bounds[0], min(bounds[1], self.alpha[i]))


def OnevsAllSVM(max_iter=math.inf, kernel=lambda x, y: x.T.dot(y), C=1.0, epsilon=0):
    return sklearn.multiclass.OneVsRestClassifier(BinarySVM(max_iter, kernel, C, epsilon))


def OnevsOneSVM(max_iter=math.inf, kernel=lambda x, y: x.T.dot(y), C=1.0, epsilon=0):
    return sklearn.multiclass.OneVsOneClassifier(BinarySVM(max_iter, kernel, C, epsilon))


def SVM(max_iter=math.inf, kernel=lambda x, y: x.T.dot(y), C=1.0, epsilon=0):
    return OnevsAllSVM(max_iter, kernel, C, epsilon)
