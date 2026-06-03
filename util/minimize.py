"""
minimize.py - Faithful line-by-line translation of MATLAB minimize.m (Carl Edward Rasmussen).
Minimizes a smooth differentiable multivariate function using BFGS or CG.

Usage: [X, fX, i, p] = minimize(X, F, p, *args)
"""
import numpy as np
import sys
from .unwrap import unwrap
from .rewrap import rewrap


_INT = 0.1   # interpolate limit for minCubic
_EXT = 5.0   # extrapolate limit for minCubic


def minimize(X, F, p, *args):
    """Main entry point, matches MATLAB minimize() lines 41-65."""
    if isinstance(p, (int, float)):
        p = {'length': p}
    else:
        p = dict(p)  # copy

    if p.get('length', 0) > 0:
        p['S'] = 'linesearch #'
    else:
        p['S'] = 'function evaluation #'

    x = unwrap(X)
    x = np.asarray(x, dtype=float).ravel()

    if 'method' not in p:
        if len(x) > 1000:
            p['method'] = 'LBFGS'
        else:
            p['method'] = 'BFGS'

    if 'verbosity' not in p:
        p['verbosity'] = 1
    if 'MFEPLS' not in p:
        p['MFEPLS'] = 10
    if 'MSR' not in p:
        p['MSR'] = 100

    # Set up function evaluator (matches MATLAB f() lines 225-231)
    def f_eval(x_vec):
        X_rewrapped, _ = rewrap(X, x_vec.copy())
        fx_val, dfx_val = F(X_rewrapped, *args)
        return fx_val, unwrap(dfx_val)

    fx, dfx = f_eval(x)

    if not np.isscalar(fx):
        raise ValueError('Objective function value must be a scalar')
    if np.any(np.isnan(fx + dfx)) or np.any(np.isinf(fx + dfx)):
        raise ValueError('Evaluating objective function with initial parameters returns NaN or Inf')

    if p['verbosity']:
        print(f'Initial Function Value {fx:4.6e}')

    method = p['method']
    if method == 'CG':
        x, fX, i, p = _cg(x, fx, dfx, p, f_eval)
    elif method == 'BFGS':
        x, fX, i, p = _bfgs(x, fx, dfx, p, f_eval)
    elif method == 'LBFGS':
        x, fX, i, p = _lbfgs(x, fx, dfx, p, f_eval)
    else:
        raise ValueError(f'Unknown method: {method}')

    X, _ = rewrap(X, x)
    if p['verbosity']:
        print()
    return X, fX, i, p


# ================ CG (lines 67-84) ================
def _cg(x0, fx0, dfx0, p, f_eval):
    if 'SIG' not in p:
        p['SIG'] = 0.1
    i = int(p['length'] < 0)
    ok = True
    r = -dfx0
    s = -float(r @ r)
    b_step = -1.0 / (s - 1)
    bs = -1.0
    fx = [fx0]
    x = x0.copy()
    dfx = dfx0.copy()

    while i < abs(p['length']):
        b_step = b_step * bs / min(b_step * s, bs / p['MSR'])
        r_norm = np.linalg.norm(r)
        b_step = min(b_step, 1.0 / max(r_norm, 1e-16))
        b_step = max(b_step, 1e-7 / max(r_norm, 1e-16))
        x, b_step, fx0, dfx, i = _lineSearch(x, fx0, dfx, r, s, b_step, i, p, f_eval)
        if i < 0:
            i = -i
            if ok:
                ok = False
                r = -dfx
            else:
                break
        else:
            ok = True
            bs = b_step * s
            # Polack-Ribiere CG
            num = float(dfx @ (dfx - dfx0))
            den = float(dfx0 @ dfx0)
            if den != 0:
                r = num / den * r - dfx
            else:
                r = -dfx
        s = float(r @ dfx)
        if s >= 0:
            r = -dfx
            s = float(r @ dfx)
            ok = False
        x0 = x.copy()
        dfx0 = dfx.copy()
        fx.append(fx0)

    return x, np.array(fx), i, p


# ================ BFGS (lines 86-103) ================
def _bfgs(x0, fx0, dfx0, p, f_eval):
    if 'SIG' not in p:
        p['SIG'] = 0.5
    i = int(p['length'] < 0)
    ok = True
    x = x0.copy()
    fx = [fx0]
    r = -dfx0
    s = -float(r @ r)
    b_step = -1.0 / (s - 1)
    H = np.eye(len(x0))
    dfx = dfx0.copy()

    while i < abs(p['length']):
        r_norm = np.linalg.norm(r)
        b_step = min(b_step, 1.0 / max(r_norm, 1e-16))
        b_step = max(b_step, 1e-7 / max(r_norm, 1e-16))
        x, b_step, fx0, dfx, i = _lineSearch(x, fx0, dfx, r, s, b_step, i, p, f_eval)
        if i < 0:
            i = -i
            if ok:
                ok = False
            else:
                break
        else:
            ok = True
            t = x - x0
            y = dfx - dfx0
            ty = float(t @ y)
            Hy = H @ y
            if abs(ty) > 1e-16:
                yHy = float(y @ Hy)
                H = H + (ty + yHy) / (ty * ty) * np.outer(t, t) \
                    - (1.0 / ty) * np.outer(Hy, t) \
                    - (1.0 / ty) * np.outer(t, Hy)
        r = -H @ dfx
        s = float(r @ dfx)
        x0 = x.copy()
        dfx0 = dfx.copy()
        fx.append(fx0)
        p['H'] = H

    return x, np.array(fx), i, p


# ================ LBFGS (lines 105-136) ================
def _lbfgs(x0, fx0, dfx0, p, f_eval):
    if 'SIG' not in p:
        p['SIG'] = 0.5
    n = len(x0)
    k = 0
    ok = True
    x = x0.copy()
    fx = [fx0]
    bs = -1.0 / p['MSR']
    m = p.get('mem', min(100, n))
    a_mem = np.zeros(m)
    t_mem = np.zeros((n, m))
    y_mem = np.zeros((n, m))
    rho_mem = np.zeros(m)
    i = int(p['length'] < 0)
    dfx = dfx0.copy()

    while i < abs(p['length']):
        q = dfx.copy()
        # First loop: backward
        for jj in range(min(k, m)):
            j = (k - 1 - jj) % m
            a_mem[j] = float(t_mem[:, j] @ q) / max(rho_mem[j], 1e-16)
            q = q - a_mem[j] * y_mem[:, j]

        if k == 0:
            qn = float(q @ q)
            r = -q / max(qn, 1e-16)
        else:
            j = (k - 1) % m
            denom = float(y_mem[:, j] @ y_mem[:, j])
            r = -float(t_mem[:, j] @ y_mem[:, j]) / max(denom, 1e-16) * q

        # Second loop: forward
        for jj in range(min(k, m)):
            j = (k - min(k, m) + jj) % m
            beta = float(y_mem[:, j] @ r) / max(rho_mem[j], 1e-16)
            r = r - t_mem[:, j] * (a_mem[j] + beta)

        s = float(r @ dfx)
        if s >= 0:
            r = -dfx
            s = float(r @ dfx)
            k = 0
            ok = False

        b_step = bs / min(bs, s / p['MSR'])
        r_norm = np.linalg.norm(r)
        b_step = min(b_step, 1.0 / max(r_norm, 1e-16))
        b_step = max(b_step, 1e-7 / max(r_norm, 1e-16))

        if np.any(np.isnan(r)) or np.any(np.isinf(r)):
            i = -i
        else:
            x, b_step, fx0, dfx, i = _lineSearch(x, fx0, dfx, r, s, b_step, i, p, f_eval)

        if i < 0:
            i = -i
            if ok:
                ok = False
                k = 0
            else:
                break
        else:
            j = k % m
            t_mem[:, j] = x - x0
            y_mem[:, j] = dfx - dfx0
            rho_mem[j] = float(t_mem[:, j] @ y_mem[:, j])
            ok = True
            k = k + 1
            bs = b_step * s

        x0 = x.copy()
        dfx0 = dfx.copy()
        fx.append(fx0)

    return x, np.array(fx), i, p


# ================ lineSearch (lines 138-199) ================
def _lineSearch(x0, f0, df0, d, s, a, i, p, f_eval):
    """Faithful translation of MATLAB lineSearch (lines 138-199)."""
    if p['length'] < 0:
        LIMIT = min(p['MFEPLS'], -i - p['length'])
    else:
        LIMIT = p['MFEPLS']

    # p0 and p1
    p0 = {'x': 0.0, 'f': f0, 'df': df0, 's': s}
    p1 = dict(p0)
    j = 0
    p3 = {'x': a}
    _wp_setup(p0, p.get('SIG', 0.5), 0)  # set up Wolfe-Powell conditions

    # ---- Extrapolation loop (lines 147-167) ----
    while True:
        ok = False
        while not ok and j < LIMIT:
            try:
                j = j + 1
                new_x = x0 + p3['x'] * d
                p3['f'], p3['df'] = f_eval(new_x)
                p3['s'] = float(p3['df'] @ d)
                ok = True
                if np.isnan(p3['f'] + p3['s']) or np.isinf(p3['f'] + p3['s']):
                    raise ValueError('Objective returned Inf or NaN')
            except Exception:
                if p.get('verbosity', 1) > 1:
                    print()
                    import traceback
                    traceback.print_exc()
                p3['x'] = (p1['x'] + p3['x']) / 2
                ok = False
                p3['f'] = np.nan
                p3['s'] = np.nan

        if _wp_check(p3) or j >= LIMIT:
            break

        p0 = dict(p1)
        p1 = dict(p3)
        # minCubic for extrapolation (extr=1)
        dx = p1['x'] - p0['x']
        dfdiff = p1['f'] - p0['f']
        p3['x'] = p0['x'] + _minCubic(dx, dfdiff, p0['s'], p1['s'], 1)

    # ---- Interpolation loop (lines 168-190) ----
    while True:
        if np.isnan(p3['f'] + p3['s']) or np.isinf(p3['f'] + p3['s']):
            p2 = dict(p1)
            break
        if p1['f'] > p3['f']:
            p2 = dict(p3)
        else:
            p2 = dict(p1)
        if _wp_check(p2) > 1 or j >= LIMIT:
            break

        dx = p3['x'] - p1['x']
        dfdiff = p3['f'] - p1['f']
        p2['x'] = p1['x'] + _minCubic(dx, dfdiff, p1['s'], p3['s'], 0)

        ok = False
        while not ok and j < LIMIT:
            try:
                j = j + 1
                new_x = x0 + p2['x'] * d
                p2['f'], p2['df'] = f_eval(new_x)
                p2['s'] = float(p2['df'] @ d)
                ok = True
                if np.isnan(p2['f'] + p2['s']) or np.isinf(p2['f'] + p2['s']):
                    raise ValueError('Objective returned Inf or NaN')
            except Exception:
                if p.get('verbosity', 1) > 1:
                    print()
                p2['x'] = (p1['x'] + p2['x']) / 2
                ok = False
                if j >= LIMIT:
                    p2 = dict(p1)

        wp_status = _wp_check(p2)
        if (wp_status > -1 and p2['s'] > 0) or wp_status < -1:
            p3 = dict(p2)
        else:
            p1 = dict(p2)

    x_ret = x0 + p2['x'] * d
    fx_ret = p2['f']
    df_ret = p2['df']
    a_ret = p2['x']

    # Line 192: count func evals or line searches
    if p['length'] < 0:
        i = i + j
    else:
        i = i + 1

    if p['verbosity']:
        sys.stdout.write(f'\r{p["S"]} {i:6d};  value {fx_ret:4.6e}')
        sys.stdout.flush()

    if _wp_check(p2) < 2:
        i = -i  # indicate failure

    return x_ret, a_ret, fx_ret, df_ret, i


# ================ minCubic (lines 201-210) ================
def _minCubic(x, df, s0, s1, extr):
    """Minimizer of approximating cubic. Lines 201-210."""
    A = -6 * df + 3 * (s0 + s1) * x
    B = 3 * df - (2 * s0 + s1) * x
    disc = B * B - A * s0 * x
    if disc < 0:
        disc = 0
    denom = B + np.sqrt(disc)
    if abs(denom) < 1e-16:
        z = x / 2
    else:
        z = -s0 * x * x / denom

    if extr:  # extrapolating
        if not np.isreal(z) or not np.isfinite(z) or z < x or z > x * _EXT:
            z = _EXT * x
        z = max(z, (1 + _INT) * x)
    else:  # interpolating
        if not np.isreal(z) or not np.isfinite(z) or z < 0 or z > x:
            z = x / 2
        z = min(max(z, _INT * x), (1 - _INT) * x)

    return z


# ================ Wolfe-Powell conditions (lines 212-223) ================
_wp_a = 0.0
_wp_b = 0.0
_wp_c = 0.0
_wp_sig = 0.5
_wp_rho = 0.0


def _wp_setup(pt, SIG, RHO):
    """Set up Wolfe-Powell conditions. Matches wp() called with 3 args (lines 214-215)."""
    global _wp_a, _wp_b, _wp_c, _wp_sig, _wp_rho
    _wp_a = RHO * pt['s']
    _wp_b = pt['f']
    _wp_c = -SIG * pt['s']
    _wp_sig = SIG
    _wp_rho = RHO


def _wp_check(pt):
    """Check Wolfe-Powell conditions. Matches wp() called with 1 arg (lines 216-222)."""
    global _wp_a, _wp_b, _wp_c
    if pt['f'] > _wp_a * pt['x'] + _wp_b:
        if _wp_a > 0:
            return -1
        else:
            return -2
    else:
        if pt['s'] < -_wp_c:
            return 0
        elif pt['s'] > _wp_c:
            return 1
        else:
            return 2
