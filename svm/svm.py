import math
import numpy as np
import random

import sklearn.base
import sklearn.multiclass


def random_int_except(m, M, ex):
    return random.choice(
        [n for n in range(m, M+1) if n not in ex]
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

    def __init__(self, max_iter=math.inf, kernel=lambda x, y: x.T.dot(y), C=1.0, epsilon=0):
        self.max_iter = max_iter
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon

        self.n, self.d = (0, 0)
        self.K = np.zeros((n, n))
        self.alpha = np.zeros(n)
        self.w = np.zeros(n)
        self.b = 0
        self.error = math.inf
    
    def train_iters(self):
        iteration = 0
        while iteration < self.max_iter and self.error > self.epsilon:
            iteration += 1
            yield iteration

    def compute_separator(self, X, y, sample_weight=None):
        self.w = np.dot(self.alpha * y, X)
        self.b = np.mean(y - np.dot(w.T, X.T))

    def score(self, X, y, sample_weight=None):
        pass
    
    def update_couple(self, i, j, X, y):
        self.eta(i, j, X, y)
        if self.K[(i, j)] == 0:
            pass
        else:
            delta = self.delta(X, y, i, j)
            self.alpha[j] += delta
            self.alpha[i] += - y[j] * y[i] * delta
            self.clip(j, self.bounds(i, j))

    def update(self, i, X, y):
        self.update_couple(
            i,
            random_int_except(0, self.n, i),
            X,
            y
        )

    def fit(self, X, y, sample_weight=None):
        self.n, self.d = X.shape
        self.K = np.full((n, n), np.nan)
        for iteration in self.train_iters():
            map(
                lambda i: self.update(i, X, y),
                range(0, self.n)
            )
    
    def distance(self, X):
        return np.dot(self.w.T, X.T) + self.b

    def predict(self, X):
        return np.sign(self.distance(X)).astype(int) 
    
    def E(self, x, y):
        return self.distance(x) - y
    
    def eta(self, i, j, X, y):
        self.K[(i, j)] = 2 * self.kernel(X[i, :], X[j, :]) - self.kernel(X[i, :], X[i, :]) - self.kernel(X[j, :], X[j, :])

    def delta(self, X, y, i, j):
        return float(y[j] * (self.E(X[j, :], y[j]) - self.E(X[i, :], y[i]))) / self.K[(i, j)]
    
    def bounds(self, i, j):
        return (
            max(0, self.alpha[j] - (y[i] == y[j]) * self.C + y[i] * y[j] * self.alpha[i]),
            min(C, self.alpha[j] + (y[i] != y[j]) * self.C + y[i] * y[j] * self.alpha[i])
        )

    def clip(self, i, bounds):
        self.alpha[i] = max(bounds[0], min(bounds[1], self.alpha[i]))


def OnevsAllSVM(max_iter=math.inf, kernel=lambda x, y: x.T.dot(y), C=1.0, epsilon=0):
    return sklearn.multiclass.OneVsRestClassifier(BinarySVM(max_iter, kernel, C, epsilon))


def OnevsOneSVM(max_iter=math.inf, kernel=lambda x, y: x.T.dot(y), C=1.0, epsilon=0):
    return sklearn.multiclass.OneVsOneClassifier(BinarySVM(max_iter, kernel, C, epsilon))


def SVM(max_iter=math.inf, kernel=lambda x, y: x.T.dot(y), C=1.0, epsilon=0):
    return OnevsAllSVM(max_iter, kernel, C, epsilon)
