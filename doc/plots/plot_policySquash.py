# Plot: preliminary (unsquashed) policy and squashed policy
# Uses GP regression (gpr) and the squashing function (gSat).

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

_plots_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.join(_plots_dir, '..', '..', '..')
sys.path.insert(0, _plots_dir)
sys.path.insert(0, _root_dir)

from pilco_python.util.gSat import gSat
from pilco_python.gp.gpr import gpr

# Apply header styling
import header

plt.close('all')

# Set random seeds: MATLAB uses randn('state',7) and rand('state',1) separately
rng_rand = np.random.default_rng(1)   # for rand (uniform)
rng_randn = np.random.default_rng(7)  # for randn (normal)

covfunc = ('covSum', ('covSEard', 'covNoise'))  # specify ARD covariance

n = 50

Axis = [-5, 5, -1.5, 2]

# dynmodel.inputs = rand(n,1)*10 - 5  -> uniform in [-5,5]
dynmodel_inputs = rng_rand.uniform(-5, 5, size=(n, 1))

# dynmodel.hyp = [0.2, 1, 0.01]'  -> log(lengthscale), log(signal_std), log(noise_std)
dynmodel_hyp = np.array([0.2, 1.0, 0.01]).reshape(-1, 1)

# dynmodel.target = 1.5*randn(n,1)
dynmodel_target = 1.5 * rng_randn.standard_normal(size=(n, 1))

xx = np.linspace(-5, 5, 101).reshape(-1, 1)
xx2 = np.linspace(-np.pi / 2, np.pi / 2, 101)

# GP prediction: prelPolicy = gpr(dynmodel.hyp, covfunc, dynmodel.inputs, dynmodel.target, xx)
prelPolicy, _ = gpr(dynmodel_hyp.ravel(), covfunc, dynmodel_inputs, dynmodel_target, xx)

fig1 = plt.figure(1)
plt.clf()
plt.plot(xx, prelPolicy)
plt.plot(xx, np.ones_like(xx), 'k--')
plt.plot(xx, -np.ones_like(xx), 'k--')
plt.xlabel('$x$')
plt.ylabel(r'$\tilde\pi(x)$')
plt.axis(Axis)
fig1.set_size_inches(12, 6)
plt.tight_layout(pad=0.1)
from print_pdf import print_pdf
print_pdf('../figures/preliminary_policy')

# Compute squashed values using gSat with deterministic input (v=0)
squashedPolicy = np.zeros(len(xx))
sqashingFct = np.zeros(len(xx2))
for i in range(len(xx)):
    M, _, _ = gSat(np.array([prelPolicy[i, 0]]), np.array([[0.0]]),
                    np.array([0]), compute_derivatives=False)
    squashedPolicy[i] = M[0]
for i in range(len(xx2)):
    M, _, _ = gSat(np.array([xx2[i]]), np.array([[0.0]]),
                    np.array([0]), compute_derivatives=False)
    sqashingFct[i] = M[0]

fig2 = plt.figure(2)
plt.clf()
plt.plot(xx, squashedPolicy)
plt.plot(xx, np.ones_like(xx), 'k--')
plt.plot(xx, -np.ones_like(xx), 'k--')
plt.xlabel('$x$')
plt.ylabel(r'$\pi(x)$')
plt.axis(Axis)
fig2.set_size_inches(12, 6)
plt.tight_layout(pad=0.1)
print_pdf('../figures/squashed_preliminary_policy')
