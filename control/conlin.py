# conlin.py
# *Summary:* Affine controller u = Wx + b with input dimension D and
# control dimension E.
# Compute mean and covariance of the control distribution p(u) from a
# Gaussian distributed input x~N(x|m,s).
# Moreover, the inv(s)*cov(x,u) is computed.
#
#
#   M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds, dMdp, dSdp, dVdp = conlin(policy, m, s)
#
#
# *Input arguments:*
#
#   policy    policy structure
#     .p      parameters that are modified during training
#       .w    linear weights                                         [ E  x  D ]
#       .b    biases/offset                                          [ E       ]
#   m         mean of state distribution                             [ D       ]
#   s         covariance matrix of state distribution                [ D  x  D ]
#
# *Output arguments:*
#
#   M         mean of predicted control                              [ E       ]
#   S         variance of predicted control                          [ E  x  E ]
#   V         inv(s) times input-output covariance                   [ D  x  E ]
#   dMdm      deriv. of mean control wrt input mean                  [ E  x  D ]
#   dSdm      deriv. of control covariance wrt input mean            [E*E x  D ]
#   dVdm      deriv. of V wrt input mean                             [D*E x  D ]
#   dMds      deriv. of mean control wrt input covariance            [ E  x D*D]
#   dSds      deriv. of control covariance wrt input covariance      [E*E x D*D]
#   dVds      deriv. of V wrt input covariance                       [D*E x D*D]
#   dMdp      deriv. of mean control wrt policy parameters           [ E  x  P ]
#   dSdp      deriv. of control covariance wrt policy parameters     [E*E x  P ]
#   dVdp      deriv. of V wrt policy parameters                      [D*E x  P ]
#
# where P = (D+1)*E is the total number of policy parameters
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2012-07-03

import numpy as np


def conlin(policy, m, s, compute_derivatives=True):
    # 1. Extract policy parameters from policy structure
    w = np.asarray(policy['p']['w'])
    b = np.asarray(policy['p']['b'])
    b = b.ravel()
    E, D = w.shape

    # 2. Predict control signal
    m_vec = m.ravel()
    M = (w @ m_vec + b).ravel()
    S = w @ s @ w.T
    S = (S + S.T) / 2
    V = w.T

    if not compute_derivatives:
        return M, S, V

    # 3. Compute derivatives
    dMdm = w
    dSdm = np.zeros((E * E, D))
    dVdm = np.zeros((D * E, D))
    dMds = np.zeros((E, D * D))
    dSds = np.kron(w, w)
    dVds = np.zeros((D * E, D * D))

    # Symmetrize dSds
    XD = np.arange(D * D).reshape(D, D, order='F')
    XDt = XD.T.flatten(order='F')
    dSds = (dSds + dSds[:, XDt]) / 2

    XE = np.arange(E * E).reshape(E, E, order='F')
    XEt = XE.T.flatten(order='F')
    dSds = (dSds + dSds[XEt, :]) / 2

    # wTdw: permutation matrix for W'⊗W → W⊗W'
    # MATLAB: reshape(permute(reshape(eye(E*D),[E D E D]),[2 1 3 4]),[E*D E*D])
    I_ED = np.eye(E * D)
    I_ED_r = I_ED.reshape(E, D, E, D, order='F')
    wTdw = I_ED_r.transpose(1, 0, 2, 3)  # [D, E, E, D]
    wTdw = wTdw.reshape(D * E, E * D, order='F')

    P = E * D + E  # total number of policy parameters

    # dMdp = [eye(E), kron(m', eye(E))]
    dMdp = np.zeros((E, P))
    dMdp[:, :E] = np.eye(E)
    dMdp[:, E:] = np.kron(m.reshape(1, -1), np.eye(E))

    # dSdp = [zeros(E*E, E), kron(eye(E), w*s)*wTdw + kron(w*s, eye(E))]
    ws = w @ s
    dSdp = np.zeros((E * E, P))
    dSdp[:, E:] = np.kron(np.eye(E), ws) @ wTdw + np.kron(ws, np.eye(E))
    dSdp = (dSdp + dSdp[XEt, :]) / 2

    # dVdp = [zeros(D*E, E), wTdw]
    dVdp = np.zeros((D * E, P))
    dVdp[:, E:] = wTdw

    return M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds, dMdp, dSdp, dVdp
