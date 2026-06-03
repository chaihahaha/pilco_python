# valueT.py
# *Summary:* Test derivatives of the propagate function, which computes the
# mean and the variance of the successor state distribution, assuming that the
# current state is Gaussian distributed with mean m and covariance matrix
# s.
#
#   [d dy dh] = valueT(p, delta, m, s, dynmodel, policy, plant, cost, H)
#
#
# *Input arguments:*
#
#   p          policy parameters (can be a structure)
#     .<>      fields that contain the policy parameters (nothing else)
#   m          mean of the input distribution
#   s          covariance of the input distribution
#   dynmodel   GP dynamics model (structure)
#   policy     policy structure
#   plant      plant structure
#   cost       cost structure
#   H          prediction horizon
#   delta      (optional) finite difference parameter. Default: 1e-4
#
#
# *Output arguments:*
#
#   dd         relative error of analytical vs. finite difference gradient
#   dy         analytical gradient
#   dh         finite difference gradient
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-21

from pilco_python.test.checkgrad import checkgrad
from pilco_python.base.value import value


def valueT(p=None, m=None, s=None, dynmodel=None, policy=None, plant=None,
           cost=None, H=None, delta=None):
    ## Code

    if delta is None:
        delta = 1e-4
    if H is None:
        H = 4

    # call checkgrad directly
    d, dy, dh = checkgrad(value, p, delta, m, s, dynmodel, policy, plant, cost, H)

    return d, dy, dh
