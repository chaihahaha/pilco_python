# covNoise.py
# Independent covariance function, ie "white noise", with specified variance.
# The covariance function is specified as:
#
# k(x^p,x^q) = s2 * \delta(p,q)
#
# where s2 is the noise variance and \delta(p,q) is a Kronecker delta function
# which is 1 iff p=q and zero otherwise. The hyperparameter is
#
# logtheta = [ log(sqrt(s2)) ]
#
# (C) Copyright 2006 by Carl Edward Rasmussen, 2006-03-24.

import numpy as np


def covNoise(loghyper, x, z=None):
    if loghyper is None and x is None:
        return '1'  # report number of parameters

    s2 = np.exp(2 * loghyper)  # noise variance (loghyper is scalar)

    if z is None:                                      # compute covariance matrix
        A = s2 * np.eye(x.shape[0])
        return A
    elif isinstance(z, np.ndarray):                             # compute test set covariances
        A = s2
        B = 0                               # zeros cross covariance by independence
        return A, B
    else:                                                # compute derivative matrix
        A = 2 * s2 * np.eye(x.shape[0])
        return A
