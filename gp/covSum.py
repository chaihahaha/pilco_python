# covSum.py
# *Summary:* Compose a covariance function as the sum of other covariance
# functions. This function doesn't actually compute very much on its own, it
# merely does some bookkeeping, and calls other covariance functions to do the
# actual work.
#
# (C) Copyright 2006 by Carl Edward Rasmussen, 2006-03-20.

import numpy as np


# Registry mapping covariance function names to their implementations
_cov_registry = {}


def _get_cov_func(name):
    """Get covariance function by name."""
    if name in _cov_registry:
        return _cov_registry[name]
    # Late import to avoid circular dependencies
    if name == 'covSEard':
        from .covSEard import covSEard
        _cov_registry[name] = covSEard
    elif name == 'covNoise':
        from .covNoise import covNoise
        _cov_registry[name] = covNoise
    elif name == 'covSum':
        from .covSum import covSum
        _cov_registry[name] = covSum
    else:
        raise ValueError(f'Unknown covariance function: {name}')
    return _cov_registry[name]


def covSum(covfunc, loghyper, x, z=None):
    # covfunc is a tuple/list of covariance function names, e.g., ('covSEard',)
    # or for nested: ('covSum', ('covSEard', 'covNoise'))
    if not isinstance(covfunc, (list, tuple)):
        covfunc = [covfunc]

    # Build the function list and collect parameter count strings (like MATLAB j(i))
    sub_funcs = []
    j = []  # parameter count expression strings
    for name in covfunc:
        f = _get_cov_func(name)
        sub_funcs.append(f)
        j.append(f(None, None))  # get param count string, e.g., '(D+1)' or '1'

    if loghyper is None and x is None:                                # report number of parameters
        params_str = '+'.join(str(s) for s in j)
        return params_str

    n, D = x.shape

    # Evaluate parameter count strings now that D is known (like MATLAB eval(char(j(i))))
    nparams = [int(eval(s, {"__builtins__": {}}, {"D": D})) for s in j]

    # Build v vector: which covariance function each parameter belongs to
    v = np.concatenate([np.full(nparams[i], i, dtype=int) for i in range(len(sub_funcs))])

    if z is None:                                              # compute covariance matrix (nargin == 3 in MATLAB)
        A = np.zeros((n, n))
        for i, f in enumerate(sub_funcs):
            A = A + f(loghyper[v == i], x)
        return A
    elif isinstance(z, np.ndarray):                             # compute test set covariances (nargin == 4, nargout == 2)
        A = np.zeros(z.shape[0])
        B = np.zeros((x.shape[0], z.shape[0]))
        for i, f in enumerate(sub_funcs):
            AA, BB = f(loghyper[v == i], x, z)
            A = A + AA
            B = B + BB
        return A, B
    else:                                                # compute derivative matrix (nargin == 4, nargout == 1)
        # z is the parameter index in the full loghyper vector
        param_idx = int(z)
        i = v[param_idx]                                    # which covariance function
        j = np.sum(v[:param_idx + 1] == i) - 1                  # which parameter in that covariance function
        f = sub_funcs[i]
        A = f(loghyper[v == i], x, j)                                   # compute derivative
        return A
