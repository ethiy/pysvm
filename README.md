# pysvm

Pythonic sequential minimization optimization implementation [1] for Support Vector Machines.

## YAL

If what you are looking for is a good fast SVM library, you should look for [`LIBSVM`](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) and [`LIBLINEAR`](https://www.csie.ntu.edu.tw/~cjlin/liblinear/). Both are wrapped in the `scikit-learn` [svm](http://scikit-learn.org/stable/modules/svm.html#svm) module. The main drawback of these wrappers is the fact that it necessitates that custom kernels be precomputed. This does not take into account one of sparcity being the most interesting features of non linear SVM. In fact, You do not need to compute the whole kernel Gramm matrix in order to train an SVM classiffier. This becomes indeed prohibitive in case of big datasets.

This library offers a SMO implementation for SVM that helps alleviating this problem. It is based on the optimized SMO algorithm according to <cite>Keerthi et al. (2001) [2]</cite>.

## References

[1]: PLATT, John. Sequential minimal optimization: A fast algorithm for training support vector machines. 1998.

[2]: KEERTHI, S.. Sathiya , SHEVADE, Shirish Krishnaj , BHATTACHARYYA, Chiranjib, et al. Improvements to Platt's SMO algorithm for SVM classifier design. Neural computation, 2001, vol. 13, no 3, p. 637-649.