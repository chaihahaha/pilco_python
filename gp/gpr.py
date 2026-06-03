# gpr.py
# *Summary:* Gaussian process regression, with a named covariance function. Two
# modes are possible: training and prediction: if no test data are given, the
# function returns minus the log likelihood and its partial derivatives with
# respect to the hyperparameters; this mode is used to fit the hyperparameters.
# If test data are given, then (marginal) Gaussian predictions are computed,
# whose mean and variance are returned. Note that in cases where the covariance
# function has noise contributions, the variance returned in S2 is for noisy
# test targets; if you want the variance of the noise-free latent function, you
# must substract the noise variance.
#
# usage: [nlml, dnlml] = gpr(logtheta, covfunc, x, y)
#    or: [mu, S2]  = gpr(logtheta, covfunc, x, y, xstar)
#
# where:
#
#   logtheta is a (column) vector of log hyperparameters
#   covfunc  is the covariance function specification, e.g., ('covSEard',)
#            or ('covSum', ('covSEard', 'covNoise'))
#   x        is a n by D matrix of training inputs
#   y        is a (column) vector (of size n) of targets
#   xstar    is a nn by D matrix of test inputs
#   nlml     is the returned value of the negative log marginal likelihood
#   dnlml    is a (column) vector of partial derivatives of the negative
#                 log marginal likelihood wrt each log hyperparameter
#   mu       is a (column) vector (of size nn) of prediced means
#   S2       is a (column) vector (of size nn) of predicted variances
#
# (C) Copyright 2006 by Carl Edward Rasmussen (2006-03-20).

import numpy as np
from ..util.solve_chol import solve_chol
from .covSum import covSum


def gpr(logtheta, covfunc, x, y, xstar=None):
    # Determine covariance function entry point
    if isinstance(covfunc, tuple) and covfunc[0] == 'covSum':
        cov = covSum
        cov_args = covfunc[1] if len(covfunc) > 1 else ()
    else:
        cov = covSum
        cov_args = (covfunc,) if isinstance(covfunc, str) else tuple(covfunc)

    n, D = x.shape

    # Verify parameter count matches covariance function
    param_expr = cov(cov_args, None, None)  # returns param count expression string
    expected_params = int(eval(param_expr, {"__builtins__": {}}, {"D": D}))
    if expected_params != logtheta.shape[0]:
        raise ValueError('Error: Number of parameters do not agree with covariance function')

    K = cov(cov_args, logtheta, x)    # compute training set covariance matrix

    L = np.linalg.cholesky(K)        # cholesky factorization of the covariance (lower triangular)

    alpha = solve_chol(L.T, y)

    if xstar is None: # if no test cases, compute the negative log marginal likelihood

        out1 = 0.5 * float(y.T @ alpha) + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)

        out2 = np.zeros(logtheta.shape)       # set the size of the derivative vector
        W = solve_chol(L.T, np.eye(n)) - np.outer(alpha, alpha)  # precompute for convenience
        for i in range(out2.shape[0]):
            out2[i] = np.sum(W * cov(cov_args, logtheta, x, i)) / 2

        return out1, out2

    else:                    # ... otherwise compute (marginal) test predictions ...

        Kss, Kstar = cov(cov_args, logtheta, x, xstar)     # test covariances, Kss diag, Kstar cross

        out1 = Kstar.T @ alpha                                      # predicted means

        v = np.linalg.solve(L, Kstar)
        out2 = Kss - np.sum(v * v, axis=0)

        return out1, out2
