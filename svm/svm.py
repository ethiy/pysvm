import math
import numpy as np
import random

import sklearn.base
import sklearn.multiclass


def random_int_except(m, M, ex):
    return random.choice(
        [n for n in range(m, M) if n not in ex]
    )


def random_couple(m, M):
    return random.sample(range(m, M), k=2)


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

    def __init__(self, max_iter=math.inf, kernel=lambda x, y: x.dot(y.T), C=1.0, tolerance=0, debug=False, verbose=False, epsilon=1E-15):
        self.max_iter = max_iter
        self.kernel = kernel
        self.C = C
        self.tolerance = tolerance
        self.debug = debug
        self.verbose = verbose

        self.initialize()
        self.b = 0
        self.support_vectors_idx = []

        self.updated = 0
        self.visit_all = True

        self.error = math.inf
        self.errors = [self.error] if self.debug else None

    def initialize(self, X=[], Y=[]):
        self.n = len(X)
        self.X = X
        self.Y = Y
        self.alpha = np.zeros(self.n)

        self.K = np.full((self.n, self.n), np.nan)
        self.Fs = np.full(self.n, np.nan)
        self.eta = dict()

        self.compute_diagonal()
    
    def compute_diagonal(self):
        for i in range(self.n):
            self.K[i, i] = self.kernel(self.X[i], self.X[i])

    def is_continuing(self):
        return bool(self.updated) or self.visit_all
    
    def is_lower_bound(self, i):
        return math.isclose(self.alpha[i], O, np.finfo(float).eps)

    def is_upper_bound(self, i):
        return math.isclose(self.alpha[i], self.C, np.finfo(float).eps)

    def is_bound(self, i):
        return self.is_lower_bound(i) or self.is_upper_bound(i)

    def train_iterations(self):
        iteration = 0
        while iteration < self.max_iter and (self.is_continuing()):
            iteration += 1
            yield iteration
    
    def F(self, x, y):
        return self.phi(x) - y
    
    def compute_eta(self, i, j):
        if (min(i, j), max(i, j)) not in self.eta.keys():
            if np.isnan(self.K[i,j]):
                self.K[i, j] = self.kernel(self.X[i], self.X[j])
                self.K[j, i] = self.K[i, j]
            self.eta[(min(i, j), max(i, j))] = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
    
    def discrepancy(self, i, j):
        return float(self.Y[j] * (self.F(self.X[j], self.Y[j]) - self.F(self.X[i], self.Y[i])))

    def delta(self, dd, i, j):
        return dd / self.eta[(min(i, j), max(i, j))]
    
    def alpha_bounds(self, i, j):
        return (
            max(0, self.alpha[j] - (self.Y[i] == self.Y[j]) * self.C + self.Y[i] * self.Y[j] * self.alpha[i]),
            min(self.C, self.alpha[j] + (self.Y[i] != self.Y[j]) * self.C + self.Y[i] * self.Y[j] * self.alpha[i])
        )

    # def find_j(self):
    #     j0 = random.choice(range(self.n))
    #     j = j0
    #     while j < self.n + j0 and self.j_stepped:
    #         yield j%self.n
    #         j += 1


    # def update_b(self, sample_weight=None):
    #     pass
    
    def score(self, X, y, sample_weight=None):
        return 

    def update_a_j(self, i, j):
        L, H = self.alpha_bounds(i, j)
        if L == H:
            return 0
        else:
            a_j_old = self.alpha[j]
            dd = self.discrepancy(i, j)
            if self.eta[(min(i, j), max(i, j))] == 0:
                self.alpha[j] = L if dd * L < dd * H else H
            else:
                self.alpha[j] += self.delta(dd, i, j)
                self.clip(j, (L, H))
            return self.alpha[j] - a_j_old

    def take_step(self, i, j):
        if i != j:
            self.compute_eta(i, j)
            self.alpha[i] += - self.Y[j] * self.Y[i] * self.update_a_j(i, j)

    def visit(self, i):


    def fit(self, X, y, sample_weight=None):
        self.initialize(X, y)
        for iteration in self.train_iterations():
            old_alpha = np.copy(self.alpha)
            self.updated = 0

            self.updated = sum(
                [
                    self.visit(i)
                    for i in range(self.n)
                    if self.visit_all or not self.is_bound(i)
                ]
            )
            self.visit_all = not self.is_continuing()

            self.error = np.linalg.norm(self.alpha - prev_alpha)
            if self.debug:
                self.errors.append(self.error)
            if self.verbose:
                print(
                    'Iteration:', iteration,
                    '=> Error =' , self.error
                )
        self.update_b()
        return self
    
    def compute_support(self):
        self.support_vectors_idx = [
            i
            for (i, a) in enumerate(self.alpha)
            if a > 0 and a < self.C
        ]
    
    def phi(self, x):
        return np.sum(self.alpha * np.array(self.Y) * np.array([self.kernel(_x, x) for _x in self.X]))

    def predict(self, x):
        return np.sign(self.phi(x) - self.b).astype(int) 

    def clip(self, i, bounds):
        self.alpha[i] = max(bounds[0], min(bounds[1], self.alpha[i]))

    def w(self):
        return np.dot(self.alpha * np.array(self.Y), np.vstack(self.X))


def OnevsAllSVM(max_iter=math.inf, kernel=lambda x, y: x.T.dot(y), C=1.0, tolerance=0):
    return sklearn.multiclass.OneVsRestClassifier(BinarySVM(max_iter, kernel, C, tolerance))


def OnevsOneSVM(max_iter=math.inf, kernel=lambda x, y: x.T.dot(y), C=1.0, tolerance=0):
    return sklearn.multiclass.OneVsOneClassifier(BinarySVM(max_iter, kernel, C, tolerance))


def SVM(max_iter=math.inf, kernel=lambda x, y: x.T.dot(y), C=1.0, tolerance=0):
    return OnevsAllSVM(max_iter, kernel, C, tolerance)
