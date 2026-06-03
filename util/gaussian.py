# gaussian.py
# *Summary:* Generate n samples from a Gaussian p(x)=N(m,S).
# Sampling is based on the Cholesky factorization of the covariance matrix S
#
#    def gaussian(m, S, n=1)
#
# *Input arguments:*
#
#   m      mean of Gaussian                                         [D x 1]
#   S      covariance matrix of Gaussian                            [D x D]
#   n      (optional) number of samples; default: n=1
#
# *Output arguments:*
#
#   x      matrix of samples from Gaussian                          [D x n]
#
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-21

import numpy as np


def gaussian(m, S, n=1):
    ## Code

    D = S.shape[0]

    # chol(S)' in MATLAB returns lower triangular L such that L @ L.T = S
    # np.linalg.cholesky(S) returns the same lower triangular L
    L = np.linalg.cholesky(S)
    Z = np.random.randn(D, n)

    # m(:) in MATLAB reshapes m to a column vector
    # bsxfun(@plus, m(:), ...) adds mean to each column (broadcasting in numpy)
    x = m.reshape(-1, 1) + L @ Z
    return x
