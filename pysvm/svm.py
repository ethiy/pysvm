import math

import sklearn.base
import sklearn.multiclass


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

        self.alpha = np.array([])
        self.w = np.array([])
        self.b = 0
        self.error = math.inf
    
    def train_iters(self):
        while iteration < self.max_iter and self.error > self.epsilon:
            iteration += 1
            yield iteration

    def compute_separator(self, X, y, sample_weight=None):
        self.w = np.dot(self.alpha * y, X)
        self.b = np.mean(y - np.dot(w.T, X.T))

    def score(self, X, y, sample_weight=None):
        pass

    def get_params(self, deep=True):
        pass

    def set_params(self, **params):
        pass
    
    def fit(self, X, y, sample_weight=None):
        for 

    def predict(self, X):
        return np.sign(np.dot(self.w.T, X.T) + self.b).astype(int) 
    

def OnevsAllSVM(max_iter=math.inf, kernel=lambda x, y: x.T.dot(y), C=1.0, epsilon=0):
    return sklearn.multiclass.OneVsRestClassifier(BinarySVM(max_iter, kernel, C, epsilon))


def OnevsOneSVM(max_iter=math.inf, kernel=lambda x, y: x.T.dot(y), C=1.0, epsilon=0):
    return sklearn.multiclass.OneVsOneClassifier(BinarySVM(max_iter, kernel, C, epsilon))


def SVM(max_iter=math.inf, kernel=lambda x, y: x.T.dot(y), C=1.0, epsilon=0):
    return OnevsAllSVM(max_iter, kernel, C, epsilon)
