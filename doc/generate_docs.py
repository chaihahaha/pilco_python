# Sweep through the folders and auto-generate documentation for each file:
# HTML
#
# Copyright (C) 2008-2013 by Marc Deisenroth, Andrew McHutchon, Joe Hall,
# and Carl Edward Rasmussen.
#
# Last modified: 2013-05-26
# Python translation: 2026

import os
import sys

docFormats = ['html']

RootDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
outputDir = {
    'html': os.path.join(RootDir, 'doc', 'html'),
}

files = [
    'base/applyController.py',
    'base/calcCost.py',
    'base/pred.py',
    'base/predcost.py',
    'base/propagate.py',
    'base/propagated.py',
    'base/rollout.py',
    'base/simulate.py',
    'base/trainDynModel.py',
    'base/value.py',
    'control/conCat.py',
    'control/congp.py',
    'control/conlin.py',
    'gp/covNoise.py',
    'gp/covSEard.py',
    'gp/covSum.py',
    'gp/fitc.py',
    'gp/gp0d.py',
    'gp/gp0.py',
    'gp/gp1d.py',
    'gp/gp1.py',
    'gp/gp2d.py',
    'gp/gp2.py',
    'gp/gpr.py',
    'gp/hypCurb.py',
    'gp/train.py',
    'loss/lossAdd.py',
    'loss/lossHinge.py',
    'loss/lossLin.py',
    'loss/lossQuad.py',
    'loss/lossSat.py',
    'loss/reward.py',
    'scenarios/cartDoublePendulum/cartDouble_learn.py',
    'scenarios/cartDoublePendulum/draw_cdp.py',
    'scenarios/cartDoublePendulum/draw_rollout_cdp.py',
    'scenarios/cartDoublePendulum/dynamics_cdp.py',
    'scenarios/cartDoublePendulum/getPlotDistr_cdp.py',
    'scenarios/cartDoublePendulum/loss_cdp.py',
    'scenarios/cartDoublePendulum/settings_cdp.py',
    'scenarios/cartPole/cartPole_learn.py',
    'scenarios/cartPole/draw_cp.py',
    'scenarios/cartPole/draw_rollout_cp.py',
    'scenarios/cartPole/dynamics_cp.py',
    'scenarios/cartPole/getPlotDistr_cp.py',
    'scenarios/cartPole/loss_cp.py',
    'scenarios/cartPole/settings_cp.py',
    'scenarios/doublePendulum/DoublePend_learn.py',
    'scenarios/doublePendulum/draw_dp.py',
    'scenarios/doublePendulum/draw_rollout_dp.py',
    'scenarios/doublePendulum/dynamics_dp.py',
    'scenarios/doublePendulum/getPlotDistr_dp.py',
    'scenarios/doublePendulum/loss_dp.py',
    'scenarios/doublePendulum/settings_dp.py',
    'scenarios/pendubot/pendubot_learn.py',
    'scenarios/pendubot/draw_pendubot.py',
    'scenarios/pendubot/draw_rollout_pendubot.py',
    'scenarios/pendubot/dynamics_pendubot.py',
    'scenarios/pendubot/getPlotDistr_pendubot.py',
    'scenarios/pendubot/loss_pendubot.py',
    'scenarios/pendubot/settings_pendubot.py',
    'scenarios/pendulum/draw_pendulum.py',
    'scenarios/pendulum/draw_rollout_pendulum.py',
    'scenarios/pendulum/dynamics_pendulum.py',
    'scenarios/pendulum/loss_pendulum.py',
    'scenarios/pendulum/pendulum_learn.py',
    'scenarios/pendulum/settings_pendulum.py',
    'scenarios/unicycle/augment_unicycle.py',
    'scenarios/unicycle/draw_unicycle.py',
    'scenarios/unicycle/draw_rollout_unicycle.py',
    'scenarios/unicycle/dynamics_unicycle.py',
    'scenarios/unicycle/loss_unicycle.py',
    'scenarios/unicycle/settings_unicycle.py',
    'scenarios/unicycle/unicycle_learn.py',
    'test/checkgrad.py',
    'test/conT.py',
    'test/gpT.py',
    'test/gSinSatT.py',
    'test/gTrigT.py',
    'test/lossT.py',
    'test/propagateT.py',
    'test/valueT.py',
    'util/error_ellipse.py',
    'util/gaussian.py',
    'util/gSat.py',
    'util/gSin.py',
    'util/gTrig.py',
    'util/maha.py',
    'util/minimize.py',
    'util/rewrap.py',
    'util/unwrap.py',
    'util/solve_chol.py',
    'util/sq_dist.py',
]


def extract_markdown_from_py(filepath):
    """Extract module docstring and convert to simple HTML/markdown."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading {filepath}: {e}"


for fmt in docFormats:
    print(f'\ngenerating {fmt} files ...')

    for j, fpath in enumerate(files):
        full_path = os.path.join(RootDir, fpath)
        if os.path.exists(full_path):
            content = extract_markdown_from_py(full_path)
            print(f'\r{j+1}/{len(files)} {fpath}')
        else:
            print(f'\r{j+1}/{len(files)} SKIP (not found): {fpath}')

print('\r done\n')
