# trainDynModel.py
# *Summary:* Script to learn the dynamics model
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modification: 2013-05-20
#
# High-Level Steps
# # Extract states and controls from x-matrix
# # Define the training inputs and targets of the GP
# # Train the GP

import numpy as np


def train_dyn_model(x, y, dynmodel, policy, plant, trainOpt):
    """
    Train the GP dynamics model from observed data.

    Parameters
    ----------
    x : ndarray (N, D_full)
        State trajectory data (full state dimension)
    y : ndarray (N, E)
        State differences (targets for GP)
    dynmodel : dict
        Dynamics model structure (will be updated)
    policy : dict
        Policy structure (contains maxU)
    plant : dict
        Plant structure (contains angi, dyni, dyno, difi)
    trainOpt : dict
        Training options passed to dynmodel.train

    Returns
    -------
    dynmodel : dict
        Updated dynamics model with trained hyperparameters
    """
    # Code

    # 1. Train GP dynamics model
    Du = len(policy['maxU'])
    Da = len(plant['angi'])                 # no. of ctrl and angles

    dyno = plant['dyno']                    # dynamics model output indices

    # x augmented with angles
    # MATLAB: xaug = [x(:,dyno) x(:,end-Du-2*Da+1:end-Du)];
    # This concatenates: state outputs and angle/cos/sin columns before the control columns
    xaug = np.hstack([x[:, dyno], x[:, -(Du + 2 * Da):-Du]])

    # dynmodel.inputs = [xaug(:,dyni) x(:,end-Du+1:end)];  use dyni and ctrl
    dyni = plant['dyni']
    dynmodel['inputs'] = np.hstack([xaug[:, dyni], x[:, -Du:]])

    # dynmodel.targets = y(:,dyno);
    dynmodel['targets'] = y[:, dyno]

    difi = plant['difi']
    # dynmodel.targets(:,difi) = dynmodel.targets(:,difi) - x(:,dyno(difi));
    dynmodel['targets'][:, difi] = dynmodel['targets'][:, difi] - x[:, np.array(dyno)[difi]]

    # Train dynamics GP
    dynmodel, _ = dynmodel['train'](dynmodel, plant, trainOpt)  # train dynamics GP

    # display some hyperparameters
    Xh = dynmodel['hyp']
    # noise standard deviations
    print('Learned noise std: ' + str(np.exp(Xh[-1, :])))
    # signal-to-noise ratios (values > 500 can cause numerical problems)
    print('SNRs             : ' + str(np.exp(Xh[-2, :] - Xh[-1, :])))

    return dynmodel
