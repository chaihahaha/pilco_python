# Set default matplotlib parameters for publication-quality figures.
# Equivalent to MATLAB's defaultaxesfontsize, defaulttextfontsize, etc.

import matplotlib as mpl

mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['font.size'] = 26
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['lines.linewidth'] = 2

# MATLAB-style line style order: solid, dashed, dotted, dash-dot
mpl.rcParams['axes.prop_cycle'] = mpl.rcsetup.cycler(
    'linestyle', ['-', '--', ':', '-.']
)
