# Plot the squashing function sigma(x) = (9*sin(x) + sin(3*x))/8

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

_plots_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _plots_dir)

import header
from print_pdf import print_pdf

plt.close('all')

x = np.linspace(-5 / 2 * np.pi, 5 / 2 * np.pi, 1000)

def squash(x):
    return (9 * np.sin(x) + np.sin(3 * x)) / 8

y = squash(x)

# plot a few periods
fig1 = plt.figure(1)
plt.clf()
plt.plot(x, y)
plt.xlabel('$x$')
plt.ylabel(r'$\sigma(x)$')
plt.axis('tight')

fig1.set_size_inches(12, 6)
plt.tight_layout(pad=0.1)
print_pdf('../figures/squashing_fct')

# plot a single period
plt.axis([-np.pi / 2, np.pi / 2, -1, 1])
print_pdf('../figures/squashing_fct2')
