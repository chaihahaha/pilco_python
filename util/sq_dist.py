"""
# sq_dist - a function to compute a matrix of all pairwise squared distances
# between two sets of vectors, stored in the columns of the two matrices, a
# (of size D by n) and b (of size D by m). If only a single argument is given
# or the second matrix is empty, the missing matrix is taken to be identical
# to the first.
#
# Special functionality: If an optional third matrix argument Q is given, it
# must be of size n by m, and in this case a vector of the traces of the
# product of Q' and the coordinatewise squared distances is returned.
#
# NOTE: The program code is written in the C language for efficiency and is
# contained in the file sq_dist.c, and should be compiled using matlabs mex
# facility. However, this file also contains a (less efficient) matlab
# implementation, supplied only as a help to people unfamiliar with mex. If
# the C code has been properly compiled and is avaiable, it automatically
# takes precendence over the matlab code in this file.
#
# Usage: C = sq_dist(a, b)
#    or: C = sq_dist(a)  or equiv.: C = sq_dist(a, [])
#    or: c = sq_dist(a, b, Q)
# where the b matrix may be empty.
#
# where a is of size D by n, b is of size D by m (or empty), C and Q are of
# size n by m and c is of size D by 1.
#
# Copyright (c) 2003, 2004, 2005 and 2006 Carl Edward Rasmussen. 2006-03-09.
"""

import numpy as np


def sq_dist(a, b=None, Q=None):
    # function C = sq_dist(a, b, Q);

    # if nargin < 1 | nargin > 3 | nargout > 1
    #   error('Wrong number of arguments.');
    # end
    # Python handles argument count via function signature; nargout check not needed

    # if nargin == 1 | isempty(b)                   % input arguments are taken to be
    if b is None:
        # b = a;                                   % identical if b is missing or empty
        b = a
    # end

    # [D, n] = size(a);
    D, n = a.shape
    # [d, m] = size(b);
    d, m = b.shape
    # if d ~= D
    if d != D:
        # error('Error: column lengths must agree.');
        raise ValueError('Error: column lengths must agree.')
    # end

    # if nargin < 3
    if Q is None:
        # C = zeros(n,m);
        C = np.zeros((n, m))
        # for d = 1:D
        for d_idx in range(D):
            # C = C + (repmat(b(d,:), n, 1) - repmat(a(d,:)', 1, m)).^2;
            # MATLAB d is 1-indexed; Python d_idx is 0-indexed
            C = C + (np.tile(b[d_idx:d_idx+1], (n, 1)) - np.tile(a[d_idx:d_idx+1].T, (1, m))) ** 2
        # end
        # % C = repmat(sum(a.*a)',1,m)+repmat(sum(b.*b),n,1)-2*a'*b could be used to
        # % replace the 3 lines above; it would be faster, but numerically less stable.
    # else
    else:
        # if [n m] == size(Q)
        if (n, m) == Q.shape:
            # C = zeros(D,1);
            C = np.zeros(D)
            # for d = 1:D
            for d_idx in range(D):
                # C(d) = sum(sum((repmat(b(d,:), n, 1) - repmat(a(d,:)', 1, m)).^2.*Q));
                # MATLAB d is 1-indexed; Python d_idx is 0-indexed
                C[d_idx] = np.sum((np.tile(b[d_idx:d_idx+1], (n, 1)) - np.tile(a[d_idx:d_idx+1].T, (1, m))) ** 2 * Q)
            # end
        # else
        else:
            # error('Third argument has wrong size.');
            raise ValueError('Third argument has wrong size.')
        # end
    # end

    return C
