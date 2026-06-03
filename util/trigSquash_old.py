"""
Augment a Gaussian with e*sin(x(i)), where i is a (possibly
empty) set of I indices. The optional e scaling factor is a vector of length
I. Optionally, compute derivatives of the parameters of the new Gaussian.

Copyright (C) 2007, 2008 & 2009 by Carl Edward Rasmussen, 2009-07-07.
Translated to Python.
"""

import numpy as np


def trigSquash(a, b, i, e=None):
    """
    Augment a Gaussian with e*sin(x(i)).

    Parameters
    ----------
    a : ndarray, shape (d,)
        Mean vector of length "d" of a Gaussian distribution
    b : ndarray, shape (d, d)
        Covariance matrix
    i : ndarray, shape (I,)
        Vector of length I of indices of elements to augment (0-indexed!)
    e : ndarray, shape (I,), optional
        Scale vector of length I (defaults to unity)

    Returns
    -------
    m : ndarray, shape (D,)
        Mean vector
    v : ndarray, shape (D, D)
        Covariance matrix
    dmda : ndarray, shape (D, d)
        Derivatives of m wrt a
    dmdb : ndarray, shape (D, d, d)
        Derivatives of m wrt b
    dvda : ndarray, shape (D, D, d)
        Derivatives of v wrt a
    dvdb : ndarray, shape (D, D, d, d)
        Derivatives of v wrt b

    where D = d + I.
    """
    d = len(a)
    if i is None or len(i) == 0:
        I = 0
        i = np.array([], dtype=int)
    else:
        i = np.asarray(i, dtype=int)
        I = len(i)
    D = d + I

    if e is None:
        e = np.ones(I)
    else:
        e = np.asarray(e).ravel()

    ai = a[i]  # shape (I,)
    bi = b[np.ix_(i, i)]  # shape (I, I)
    bii = np.diag(bi)  # shape (I,)

    m = np.zeros(D)
    m[:d] = a  # first the means
    if I > 0:
        m[d:] = e * np.exp(-bii / 2) * np.sin(ai)

    v = np.zeros((D, D))
    v[:d, :d] = b  # the covariance

    if I > 0:
        cross_term = b[i, :] * np.outer(e * np.exp(-bii / 2) * np.cos(ai), np.ones(d))
        v[d:, :d] = cross_term
        v[:d, d:] = cross_term.T  # symmetric entries

        for j in range(I):
            for k in range(I):
                if j == k:
                    q = e[j] * e[k] * (1 - np.exp(-bii[j])) / 2
                    v[d + j, d + j] = q * (1 + np.exp(-bii[j]) * np.cos(2 * ai[j]))
                else:
                    q = e[j] * e[k] * np.exp(-(bii[j] + bii[k]) / 2) / 2

                    # for numerical reasons:
                    logq = np.log(e[j]) + np.log(e[k]) - (bii[j] + bii[k]) / 2 - np.log(2)
                    v[d + j, d + k] = (np.exp(logq + bi[j, k]) - q) * np.cos(ai[j] - ai[k]) \
                        - (np.exp(logq - bi[j, k]) - q) * np.cos(ai[j] + ai[k])
                    v[d + k, d + j] = v[d + j, d + k]

    # compute derivatives
    dmda = np.zeros((D, d))
    dmda[:d, :] = np.eye(d)

    dmdb = np.zeros((D, d, d))

    dvda = np.zeros((D, D, d))

    dvdb = np.zeros((D, D, d, d))

    if I > 0:
        dmda[d:, i] = np.diag(e * np.exp(-bii / 2) * np.cos(ai))

        for j in range(I):
            dmdb[d + j, i[j], i[j]] = -m[d + j] / 2

        for j in range(I):
            z = e[j] * b[:, i[j]] * np.exp(-bii[j] / 2)
            dvda[d + j, :d, i[j]] = -z * np.sin(ai[j])
            dvda[:d, d + j, i[j]] = -z * np.sin(ai[j])

            for k in range(I):
                if j == k:
                    zz = e[j] * e[k] * (1 - np.exp(-bii[j])) * np.exp(-bii[j])
                    dvda[d + j, d + j, i[j]] = -zz * np.sin(2 * ai[j])
                else:
                    q = e[j] * e[k] * np.exp(-(bii[j] + bii[k]) / 2) / 2
                    logq = np.log(e[j]) + np.log(e[k]) - (bii[j] + bii[k]) / 2 - np.log(2)
                    dvda[d + j, d + k, i[j]] = -(np.exp(logq + bi[j, k]) - q) * np.sin(ai[j] - ai[k]) + \
                        (np.exp(logq - bi[j, k]) - q) * np.sin(ai[j] + ai[k])
                    dvda[d + j, d + k, i[k]] = (np.exp(logq + bi[j, k]) - q) * np.sin(ai[j] - ai[k]) + \
                        (np.exp(logq - bi[j, k]) - q) * np.sin(ai[j] + ai[k])

        for j in range(d):
            for k in range(d):
                dvdb[j, k, j, k] = 0.5
                dvdb[j, k, k, j] = dvdb[j, k, k, j] + 0.5

        for j in range(I):
            dvdb[d + j, :d, i[j], i[j]] = -v[d + j, :d] / 2
            dvdb[:d, d + j, i[j], i[j]] = -v[d + j, :d] / 2

            for k in range(I):
                # derivative of trig variance w.r.t. input variance
                if j == k:
                    # trig variance
                    q = e[j] * e[k] * np.exp(-bii[j]) / 2
                    dvdb[d + j, d + j, i[j], i[j]] = q * (1 + np.cos(2 * ai[j]) * (2 * np.exp(-bii[j]) - 1))
                else:
                    # trig-trig covariance terms
                    logq = np.log(e[j]) + np.log(e[k]) - (bii[j] + bii[k]) / 2 - np.log(2)
                    dvdb[d + j, d + k, i[j], i[k]] = 0.5 * ((np.exp(logq + bi[j, k]) * np.cos(ai[j] - ai[k]) +
                                                              np.exp(logq - bi[j, k]) * np.cos(ai[j] + ai[k])))
                    dvdb[d + k, d + j, i[j], i[k]] = dvdb[d + j, d + k, i[j], i[k]]
                    dvdb[d + j, d + k, i[j], i[j]] = -v[d + j, d + k] / 2
                    dvdb[d + j, d + k, i[k], i[k]] = -v[d + j, d + k] / 2

            z = e[j] * np.exp(-bii[j] / 2) / 2
            zz = e[j] * (1 - bii[j] / 2) * np.exp(-bii[j] / 2)
            for k in range(d):
                # derivative of covariance of trig-nontrig w.r.t input variance
                if k == i[j]:
                    dvdb[k, d + j, k, k] = zz * np.cos(ai[j])
                    dvdb[d + j, k, k, k] = zz * np.cos(ai[j])
                else:
                    dvdb[k, d + j, k, i[j]] = z * np.cos(ai[j])
                    dvdb[d + j, k, k, i[j]] = z * np.cos(ai[j])
                    dvdb[k, d + j, i[j], k] = z * np.cos(ai[j])
                    dvdb[d + j, k, i[j], k] = z * np.cos(ai[j])

    return m, v, dmda, dmdb, dvda, dvdb
