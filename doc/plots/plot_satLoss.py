# Plot the saturating cost function.

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

_plots_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.join(_plots_dir, '..', '..', '..')
sys.path.insert(0, _plots_dir)
sys.path.insert(0, _root_dir)

from pilco_python.loss.lossSat import lossSat

import header
from print_pdf import print_pdf

plt.close('all')

cost = {}
cost['fcn'] = 'lossSat'  # reference; we'll call lossSat directly
cost['z'] = np.array([[0.0]])  # target state (1 x 1)
cost['width'] = 0.5

cost['W'] = np.array([[1.0 / (cost['width'] ** 2)]])  # weight (1 x 1 matrix)

x = np.linspace(0, 3, 200)  # array of distances

c = np.zeros(len(x))

# evaluate cost for all distances x
for i in range(len(x)):
    # lossSat(cost, m, s) where m=scalar mean, s=scalar covariance
    L, _, _, _, _, _, _, _, _ = lossSat(cost, np.array([[x[i]]]), np.array([[0.0]]),
                                         compute_derivatives=False)
    c[i] = L[0, 0]

# evaluate cost at 2-sigma bound
L2, _, _, _, _, _, _, _, _ = lossSat(cost, np.array([[2 * cost['width']]]),
                                     np.array([[0.0]]), compute_derivatives=False)
c2 = L2[0, 0]

fig = plt.figure()
plt.plot(x, c)  # plot cost
plt.plot(2 * cost['width'], c2, 'ro', markersize=8)  # plot 2-sigma cost value
plt.plot(np.linspace(0, 2 * cost['width'], 100),
         c2 * np.ones(100), 'k-')  # plot horizontal line
plt.text(cost['width'] / 4, c2 + 0.05, r'$2\texttt{cost.width}$')
plt.xlabel('Distance to target')
plt.ylabel('Cost')
plt.axis([x[0], x[-1], 0, 1.01])

fig.set_size_inches(12, 6)
plt.tight_layout(pad=0.1)
print_pdf('../figures/satLossPlot')
