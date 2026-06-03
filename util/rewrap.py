# rewrap.py
# *Summary:* Map the numerical elements in the vector v onto the variables
# s, which can be of any type. The number of numerical elements must match;
# on exit, v should be empty. Non-numerical entries are just copied.
# See also the reverse unwrap.py.
#
#    (s, v) = rewrap(s, v)
#
# *Input arguments:*
#
#   s     dict (struct), list (cell), or numeric values
#   v     column vector of numerical values to map into s
#
#
# *Output arguments:*
#
#   s     dict (struct), list (cell), or numeric values (updated with v values)
#   v     [empty] remaining unused portion of v
#
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-25

import numpy as np


def rewrap(s, v):
    ## Code

    if isinstance(s, (int, float, np.integer, np.floating)):
        if len(v) < 1:
            raise ValueError('The vector for conversion contains too few elements')
        s = float(v[0])                                  # scalar recast from vector
        v = v[1:]                                        # remaining arguments passed on
    elif isinstance(s, np.ndarray):
        n = s.size                                        # number of numeric elements expected
        if len(v) < n:
            raise ValueError('The vector for conversion contains too few elements')
        s = np.array(v[:n]).reshape(s.shape, order='F')   # numeric values are reshaped to original size (Fortran order matches MATLAB)
        v = v[n:]                                          # remaining arguments passed on
    elif isinstance(s, dict):
        sorted_keys = sorted(s.keys())                     # alphabetize keys
        sorted_values = [s[k] for k in sorted_keys]        # values in alphabetical key order
        t, v = rewrap(sorted_values, v)                    # convert to list, recurse
        for i, k in enumerate(sorted_keys):               # map results back by key
            s[k] = t[i]
    elif isinstance(s, (list, tuple)):
        for i in range(len(s)):                            # list elements are handled sequentially
            s[i], v = rewrap(s[i], v)
    return s, v
