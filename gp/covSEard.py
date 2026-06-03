# covSEard.py
# Squared Exponential covariance function with Automatic Relevance Determination
# (ARD) distance measure. The covariance function is parameterized as:
#
# k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
#
# where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
# D is the dimension of the input space and sf2 is the signal variance. The
# hyperparameters are:
#
# loghyper = [ log(ell_1)
#              log(ell_2)
#               .
#              log(ell_D)
#              log(sqrt(sf2)) ]
#
# (C) Copyright 2006 by Carl Edward Rasmussen (2006-03-24)

import numpy as np
from ..util.sq_dist import sq_dist


def covSEard(loghyper, x, z=None):
    if loghyper is None and x is None:
        return '(D+1)'  # report number of parameters

    n, D = x.shape
    ell = np.exp(loghyper[:D])                         # characteristic length scale
    sf2 = np.exp(2 * loghyper[D])                                   # signal variance

    if z is None:                                            # compute covariance matrix
        K = sf2 * np.exp(-sq_dist((x / ell).T) / 2)
        return K
    elif isinstance(z, np.ndarray):                             # test set, z is array of test inputs
        A = sf2 * np.ones(z.shape[0])
        B = sf2 * np.exp(-sq_dist((x / ell).T, (z / ell).T) / 2)
        return A, B
    else:                                                # compute derivative matrix, z is parameter index
        idx = int(z)
        # Efficiently compute squared distances for one dimension
        def _single_dim_sq(a, b, d_idx):
            return (np.tile(b[:, d_idx:d_idx + 1], (a.shape[0], 1)) -
                    np.tile(a[:, d_idx:d_idx + 1].T, (1, b.shape[0]))) ** 2

        K = sf2 * np.exp(-sq_dist((x / ell).T) / 2)  # recompute or use cached
        if idx < D:                                           # length scale parameters
            K_idx = K * sq_dist(x[:, idx:idx + 1].T / ell[idx])
            return K_idx
        else:                                                    # magnitude parameter
            return 2 * K
