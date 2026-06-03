# maha.py
# Point-wise squared Mahalanobis distance (a-b)*Q*(a-b)'
# Vectors are row-vectors
#
# function K = maha(a, b, Q)
#
# Input arguments:
#   a   matrix containing n row vectors                                 [n x D]
#   b   matrix containing m row vectors                                 [m x D]
#   Q   weight matrix. Default: eye(D)                                  [D x D]
#
# Output arguments:
#   K   point-wise squared distances                                    [n x m]
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.

import numpy as np

def maha(a, b, Q=None):
    if Q is None:  # assume unit Q
        K = np.sum(a**2, axis=1, keepdims=True) + np.sum(b**2, axis=1) - 2 * a @ b.T
    else:
        aQ = a @ Q
        K = np.sum(aQ * a, axis=1, keepdims=True) + np.sum((b @ Q) * b, axis=1) - 2 * aQ @ b.T
    return K
