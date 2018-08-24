import math

import sklearn.base


class SVM(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
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

        self.alpha = None
        self.n = 0
        self.d = 0
        self.error = math.inf
        self.iteration = 0
    
    def train_iters():
        while self.iteration < self.max_iter and self.error > self.epsilon:
            self.iteration += 1
            yield True

    def score(X, y, sample_weight=None):
        pass

    def get_params(deep=True):
        pass

    def set_params(**params):
        pass
    
    def fit(X, y, sample_weight=None):
        pass

    def predict(X):
        pass 