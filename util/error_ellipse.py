"""
error_ellipse.py - plot an error ellipse, or ellipsoid, defining confidence region

ERROR_ELLIPSE(C22) - Given a 2x2 covariance matrix, plot the
associated error ellipse, at the origin. It returns a graphics handle
of the ellipse that was drawn.

ERROR_ELLIPSE(C33) - Given a 3x3 covariance matrix, plot the
associated error ellipsoid, at the origin, as well as its projections
onto the three axes. Returns a vector of 4 graphics handles, for the
three ellipses (in the X-Y, Y-Z, and Z-X planes, respectively) and for
the ellipsoid.

ERROR_ELLIPSE(C,MU) - Plot the ellipse, or ellipsoid, centered at MU,
a vector whose length should match that of C (which is 2x2 or 3x3).

ERROR_ELLIPSE(...,'Property1',Value1,'Name2',Value2,...) sets the
values of specified properties, including:
  'C' - Alternate method of specifying the covariance matrix
  'mu' - Alternate method of specifying the ellipse (-oid) center
  'conf' - A value betwen 0 and 1 specifying the confidence interval.
    the default is 0.5 which is the 50% error ellipse.
  'scale' - Allow the plot the be scaled to difference units.
  'style' - A plotting style used to format ellipses.
  'clip' - specifies a clipping radius. Portions of the ellipse, -oid,
    outside the radius will not be shown.

NOTES: C must be positive definite for this function to work properly.


by AJ Johnson
http://www.mathworks.de/matlabcentral/fileexchange/4705
Translated to Python.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2


def error_ellipse(*varargin):
    """
    Plot an error ellipse or ellipsoid from a covariance matrix.

    Parameters
    ----------
    C : ndarray, shape (2,2) or (3,3)
        Covariance matrix. Required.
    mu : ndarray, optional
        Center of ellipse. Default zeros.
    conf : float, optional
        Confidence level between 0 and 1. Default 0.95.
    scale : float, optional
        Scale factor. Default 1.
    style : str, optional
        Plot style for ellipses. Default ''.
    clip : float, optional
        Clipping radius. Default inf.

    Returns
    -------
    h : matplotlib artist or list of artists
        Graphics handles of the drawn objects.
    """
    default_properties = {
        'C': None,
        'mu': None,
        'conf': 0.95,
        'scale': 1,
        'style': '',
        'clip': np.inf,
    }

    args = list(varargin)

    if len(args) >= 1 and isinstance(args[0], np.ndarray) and args[0].dtype.kind in 'fiuc':
        default_properties['C'] = args.pop(0)

    if len(args) >= 1 and isinstance(args[0], np.ndarray) and args[0].dtype.kind in 'fiuc':
        default_properties['mu'] = args.pop(0)

    if len(args) >= 1 and isinstance(args[0], (int, float)):
        default_properties['conf'] = args.pop(0)

    if len(args) >= 1 and isinstance(args[0], (int, float)):
        default_properties['scale'] = args.pop(0)

    if len(args) >= 1 and not isinstance(args[0], str):
        raise ValueError('Invalid parameter/value pair arguments.')

    prop = _getopt(default_properties, *args)
    C = prop['C']

    if prop['mu'] is None:
        mu = np.zeros(len(C))
    else:
        mu = np.asarray(prop['mu'])

    conf = prop['conf']
    scale = prop['scale']
    style = prop['style']

    if conf <= 0 or conf >= 1:
        raise ValueError('conf parameter must be in range 0 to 1, exclusive')

    r, c = C.shape
    if r != c or (r != 2 and r != 3):
        raise ValueError(f"Don't know what to do with {r}x{c} matrix")

    x0 = mu[0]
    y0 = mu[1]

    # Compute quantile for the desired percentile
    k = np.sqrt(chi2.ppf(conf, r))  # r is the number of dimensions (degrees of freedom)

    if r == 3 and c == 3:
        z0 = mu[2]

        # Make the matrix has positive eigenvalues
        if np.any(np.linalg.eigvalsh(C) <= 0):
            raise ValueError('The covariance matrix must be positive definite (it has non-positive eigenvalues)')

        # C is 3x3; extract the 2x2 matrices, and plot the associated error ellipses
        Cxy = C[0:2, 0:2]
        Cyz = C[1:3, 1:3]
        Czx = C[[2, 0], :][:, [2, 0]]

        x, y, z = _getpoints(Cxy, prop['clip'])
        h1, = plt.plot(x0 + k * x, y0 + k * y, z0 + k * z, style if style else None, linewidth=2)
        y, z, x = _getpoints(Cyz, prop['clip'])
        h2, = plt.plot(x0 + k * x, y0 + k * y, z0 + k * z, style if style else None, linewidth=2)
        z, x, y = _getpoints(Czx, prop['clip'])
        h3, = plt.plot(x0 + k * x, y0 + k * y, z0 + k * z, style if style else None, linewidth=2)

        eigval, eigvec = np.linalg.eigh(C)

        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        X_ell = np.outer(np.cos(u), np.sin(v))
        Y_ell = np.outer(np.sin(u), np.sin(v))
        Z_ell = np.outer(np.ones_like(u), np.cos(v))

        n_pts = X_ell.size
        XYZ = np.column_stack([X_ell.ravel(), Y_ell.ravel(), Z_ell.ravel()]) @ np.sqrt(np.diag(eigval)) @ eigvec.T

        X_ell = scale * (k * XYZ[:, 0].reshape(X_ell.shape) + x0)
        Y_ell = scale * (k * XYZ[:, 1].reshape(Y_ell.shape) + y0)
        Z_ell = scale * (k * XYZ[:, 2].reshape(Z_ell.shape) + z0)

        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.gca()
        h4 = ax.plot_surface(X_ell, Y_ell, Z_ell, cmap='gray', alpha=0.3)

        return [h1, h2, h3, h4]

    elif r == 2 and c == 2:
        # Make the matrix has positive eigenvalues
        if np.any(np.linalg.eigvalsh(C) <= 0):
            raise ValueError('The covariance matrix must be positive definite (it has non-positive eigenvalues)')

        x, y, z = _getpoints(C, prop['clip'])
        h1, = plt.plot(scale * (x0 + k * x), scale * (y0 + k * y), style if style else None, linewidth=2)
        return h1

    else:
        raise ValueError('C (covariance matrix) must be specified as a 2x2 or 3x3 matrix)')


def _getpoints(C, clipping_radius=None):
    """
    Generate x and y points that define an ellipse, given a 2x2 covariance matrix, C.
    z, if requested, is all zeros with same shape as x and y.
    """
    n = 100  # Number of points around ellipse
    p = np.arange(0, 2 * np.pi + np.pi / n, np.pi / n)  # angles around a circle

    eigval, eigvec = np.linalg.eigh(C)  # Compute eigen-stuff
    xy = np.column_stack([np.cos(p), np.sin(p)]) @ np.sqrt(np.diag(eigval)) @ eigvec.T  # Transformation
    x = xy[:, 0]
    y = xy[:, 1]
    z = np.zeros_like(x)

    # Clip data to a bounding radius
    if clipping_radius is not None and np.isfinite(clipping_radius):
        r = np.sqrt(np.sum(xy ** 2, axis=1))  # Euclidean distance (distance from center)
        x[r > clipping_radius] = np.nan
        y[r > clipping_radius] = np.nan
        z[r > clipping_radius] = np.nan

    return x, y, z


def _getopt(properties, *varargin):
    """
    Process paired optional arguments as 'prop1',val1,'prop2',val2,...

    Parameters
    ----------
    properties : dict
        Initial properties dictionary
    *varargin : str, value pairs
        Pairs of property name and value to assign

    Returns
    -------
    properties : dict
        Modified properties dictionary
    """
    prop_names = list(properties.keys())
    target_field = None
    args = list(varargin)

    for arg in args:
        if target_field is None:
            if not isinstance(arg, str):
                raise ValueError('Property names must be character strings')
            if arg not in prop_names:
                raise ValueError(f"Invalid property '{arg}'; must be one of: {prop_names}")
            target_field = arg
        else:
            properties[target_field] = arg
            target_field = None

    if target_field is not None:
        raise ValueError('Property names and values must be specified in pairs.')

    return properties
