# unwrap.py
# *Summary:* Extract the numerical values from s into the column vector v.
# The variable s can be of any type, including dict (struct) and list (cell array).
# Non-numerical elements are ignored. See also the reverse rewrap.py.
#
#    v = unwrap(s)
#
# *Input arguments:*
#
#   s     dict (struct), list (cell), or numeric values
#
#
# *Output arguments:*
#
#   v     column vector of unwrapped numeric values
#
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-25

import numpy as np


def unwrap(s):
    ## Code

    v = np.array([], dtype=float)
    if isinstance(s, (int, float, np.integer, np.floating)):
        v = np.array([s], dtype=float)                # numeric scalar recast to column vector
    elif isinstance(s, np.ndarray):
        v = s.flatten(order='F').astype(float)         # numeric values are recast to column vector (Fortran order matches MATLAB)
    elif isinstance(s, dict):
        sorted_keys = sorted(s.keys())                 # alphabetize keys
        for k in sorted_keys:
            v = np.concatenate([v, unwrap(s[k])])      # recurse on each field value
    elif isinstance(s, (list, tuple)):
        for i in range(len(s)):                        # list elements are handled sequentially
            v = np.concatenate([v, unwrap(s[i])])
    return v
