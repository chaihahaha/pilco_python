# draw_rollout_unicycle.py
# *Summary:* Script to draw a rollout of the unicycle
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-04-04
#
# High-Level Steps
# # For each time step, plot the observed trajectory, the applied torques,
# and the incurred cost


def draw_rollout_unicycle(ax, xx, plant, dt, cost, j, J_val, H, H_val=None,
                           x=None, jj=None):
    """
    Draw a rollout of the unicycle.

    Parameters are passed from the calling context (matching MATLAB's
    unicycle_learn.m script variables).
    """
    from .draw_unicycle import draw_unicycle

    if j > J_val:
        if H_val is None:
            H_val = H
        text1 = 'trial # %d, T=%.2f sec' % (j + J_val, H_val * dt)
        if x is not None:
            text2 = 'total experience (after this trial): %.2f sec' % (dt * x.shape[0])
        else:
            text2 = ''
        draw_unicycle(xx, plant, dt / 10.0, cost, text1, text2)
    else:
        text1 = '(random) trial # %d, T=%.2f sec' % (jj if jj is not None else j, H * dt)
        if x is not None:
            text2 = 'total experience (after this trial): %.2f sec' % (dt * x.shape[0])
        else:
            text2 = ''
        draw_unicycle(xx, plant, dt / 10.0, cost, text1, text2)
