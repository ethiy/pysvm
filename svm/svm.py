from enum import Enum

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


class Bound(Enum):
    low = 1
    up = 2


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
        self.b_up, self.b_low = (0, 0)
        self.i_up, self.i_low = (0, 0)
        self.L, self.H = (0, 0)
        self.delta_i, self.delta_j = (0, 0)

        self.eta = dict()

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
        self.bounded = self.n * [Bound.low]

        self.compute_diagonal()
    
    def compute_diagonal(self):
        for i in range(self.n):
            self.K[i, i] = self.kernel(self.X[i], self.X[i])

    def is_continuing(self):
        return bool(self.updated) or self.visit_all

    def bound(self, i):
        if math.isclose(self.alpha[i], O, np.finfo(float).eps):
            return Bound.low
        elif math.isclose(self.alpha[i], self.C, np.finfo(float).eps):
            return Bound.up
        else:
            return None
    
    def train_iterations(self):
        iteration = 0
        while iteration < self.max_iter and (self.is_continuing()):
            iteration += 1
            yield iteration
    
    def F_(self, i):
        return self.phi_(i) - self.Y[i]

    def F(self, x, y):
        return self.phi(x) - y

    def sparse_kernel(self, i, x):
        return 0 if self.is_I1(i) or self.is_I4(i) else self.kernel(self.X[i], x)

    def sparse_K_(self, i, j):
        return 0 if self.is_I1(j) or self.is_I4(j) else self.K_(j, i)

    def phi_(self, i):
        return np.sum(self.alpha * np.array(self.Y) * np.array([self.sparse_K_(i, j) for j in range(self.n)]))

    def phi(self, x):
        return np.sum(self.alpha * np.array(self.Y) * np.array([self.sparse_kernel(i, x) for i in range(self.n)]))

    def K_(self, i, j):
        if np.isnan(self.K[i,j]):
            self.K[i, j] = self.kernel(self.X[i], self.X[j])
            self.K[j, i] = self.K[i, j]
        return self.K[i, j]
    
    def compute_eta(self, i, j):
        if (min(i, j), max(i, j)) not in self.eta.keys():
            self.eta[(min(i, j), max(i, j))] = 2 * self.K_(i, j) - self.K_(i, i) - self.K_(j, j)
    
    def discrepancy(self, i, j):
        return float(self.Y[j] * (self.Fs[j] - self.Fs[i]))

    def delta(self, dd, i, j):
        return dd / self.eta[(min(i, j), max(i, j))]
    
    def alpha_bounds(self, i, j):
        self.L = max(0, self.alpha[j] - (self.Y[i] == self.Y[j]) * self.C + self.Y[i] * self.Y[j] * self.alpha[i])
        self.H = min(self.C, self.alpha[j] + (self.Y[i] != self.Y[j]) * self.C + self.Y[i] * self.Y[j] * self.alpha[i])

    def clip(self, i):
        self.alpha[i] = max(self.L, min(self.H, self.alpha[i]))

    # def update_b(self, sample_weight=None):
    #     pass

    def update_a_j(self, i, j, L, H):
        a_j_old = self.alpha[j]
        dd = self.discrepancy(i, j)
        if self.eta[(min(i, j), max(i, j))] == 0:
            self.alpha[j] = L if dd * L < dd * H else (H if dd * L > dd * H else a_j_old)
        else:
            self.alpha[j] += self.delta(dd, i, j)
            self.clip(j)
        return self.alpha[j] - a_j_old

    def update_F_i_j(self, i, j):
        for k in [i,j]:
            self.Fs[k] += self.Y[i] * self.delta_i * self.K_(i, k) + self.Y[j] * self.delta_j * self.K_(j, k)
    
    def update_F_I0(self):
        for k in range(self.n):
            if self.is_I0(k):
                self.Fs[k] = self.F_(k)
    
    def update_bounded(self):
        self.bounded = np.array(
            [self.bound(i) for i in range(self.n)]
        )
    
    def update_up_low(self, i, j):
        up = [k for k in range(self.n) if self.is_I0(k)]
        low = up
        for k in [i, j]:
            if self.is_I1(k) or self.is_I2(k):
                up = up + [k]
            elif self.is_I3(k) or self.is_I4(k):
                low = low + [k]
        self.i_up, self.b_up = min(
            [(k, self.Fs[k]) for k in set(up)],
            key=lambda f: f[1]
        )
        self.i_low, self.b_low = max(
            [(k, self.Fs[k]) for k in set(low)],
            key=lambda f: f[1]
        )
    
    def step_updates(self, i, j):
        self.update_F_I0()
        self.update_bounded()
        self.update_F_i_j(i, j)
        self.update_up_low(i, j)

    def take_step(self, i, j):
        if i == j:
            return 0

        self.alpha_bounds(i, j)
        if self.L == self.H:
            return 0

        self.compute_eta(i, j)
        self.delta_j = self.update_a_j(i, j)
        if math.isclose(self.delta_j, 0, self.epsilon) < 0:
            return 0
        self.delta_i = - self.Y[j] * self.Y[i] * self.delta_j
        self.alpha[i] += self.delta_i

        self.step_updates(i, j)
        return 1
    
    def is_I0(self, j):
        return self.bounded[j] is None
    
    def is_I1(self, j):
        return self.bounded[j] == Bound.low and self.Y[j] > 0

    def is_I2(self, j):
        return self.bounded[j] == Bound.up and self.Y[j] < 0

    def is_I3(self, j):
        return self.bounded[j] == Bound.up and self.Y[j] > 0
    
    def is_I4(self, j):
        return self.bounded[j] == Bound.low and self.Y[j] < 0
    
    def update_bounds(self, j):
        self.bounded[j] = self.bound(j)
        self.Fs[j] = self.F_(j)
        if not self.is_I0(j):
            if (self.is_I1(j) or self.is_I2(j))  and self.Fs[j] < self.b_up:
                self.b_up = self.Fs[j]
                self.i_up = j
            elif (self.is_I3(j) or self.is_I4(j)) and self.Fs[j] > self.b_low:
                self.b_low = self.Fs[j]
                self.i_low = j
    
    def check_optimality(self, j):
        if (self.is_I0(j) or self.is_I1(j) or self.is_I2(j)) and (self.b_low - self.Fs[j] > 2 * self.tolerance):
            return self.i_low
        elif (self.is_I0(j) or self.is_I3(j) or self.is_I4(j)) and (self.Fs[j] - self.b_up > 2 * self.tolerance)
            return self.i_up
        else:
            return None

    def visit(self, j):
        self.update_bounds(j)
        
        i = self.check_optimality(j)
        if i is None:
            return 0
        
        if self.is_I0(j):
            i = self.i_low if self.b_low - self.Fs[j] > self.Fs[j] - self.b_up else self.i_up

        return self.take_step(i, j)

    def fit(self, X, y, sample_weight=None):
        self.initialize(X, y)
        for iteration in self.train_iterations():
            old_alpha = np.copy(self.alpha)
            self.updated = 0

            self.updated = sum(
                [
                    self.visit(i)
                    for i in range(self.n)
                    if self.visit_all or (self.bound(i) is None)
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
    
    def predict(self, x):
        return np.sign(self.phi(x) - self.b).astype(int) 

    def w(self):
        return np.dot(self.alpha * np.array(self.Y), np.vstack(self.X))


def OnevsAllSVM(max_iter=math.inf, kernel=lambda x, y: x.T.dot(y), C=1.0, tolerance=0):
    return sklearn.multiclass.OneVsRestClassifier(BinarySVM(max_iter, kernel, C, tolerance))


def OnevsOneSVM(max_iter=math.inf, kernel=lambda x, y: x.T.dot(y), C=1.0, tolerance=0):
    return sklearn.multiclass.OneVsOneClassifier(BinarySVM(max_iter, kernel, C, tolerance))


def SVM(max_iter=math.inf, kernel=lambda x, y: x.T.dot(y), C=1.0, tolerance=0):
    return OnevsAllSVM(max_iter, kernel, C, tolerance)
