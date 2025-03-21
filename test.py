from __future__ import annotations
import itertools
import random

import numpy as np
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils import draw_sobol_samples
from gpytorch import ExactMarginalLogLikelihood
import warnings

from dev.constants.bayes import MAX_STRETCHED_MUSCLE_LENGTH_ONE
from dev.constants.bayes import MAX_STRETCHED_MUSCLE_LENGTH_TWO
from dev.constants.bayes import MIN_STRETCHED_MUSCLE_LENGTH_ONE
from dev.constants.bayes import MIN_STRETCHED_MUSCLE_LENGTH_TWO
from dev.constants.bayes import NUM_INITIAL_POINTS
from dev.constants.bayes import NUM_ITERATIONS
from dev.constants.bayes import NUM_NEW_CANDIDATES
from dev.constants.bayes import SEED
from dev.constants.physical import RELAXED_MUSCLE_LENGTH_ONE
from dev.constants.physical import RELAXED_MUSCLE_LENGTH_TWO
from dev.models.hill_type_model_wrapper import HillTypeModelWrapper
from dev.visual.range_of_motion_plotter import RangeOfMotionPlotter

torch.manual_seed(SEED)


def acq_func(gaussian_process):
    return UpperConfidenceBound(gaussian_process, beta=0.1)


def gp_process(x, y):
    return SingleTaskGP(x, y)

num_vals = 3
params = {
    'Length_Slack_M1': np.linspace(RELAXED_MUSCLE_LENGTH_ONE, RELAXED_MUSCLE_LENGTH_ONE*1.2, num_vals),
    'Length_Slack_M2': np.linspace(RELAXED_MUSCLE_LENGTH_TWO, RELAXED_MUSCLE_LENGTH_TWO*1.2, num_vals),
    # 'Mass_M1' : np.linspace(0.8, 1.2, num_vals),
    # 'Mass_M2': np.linspace(0.8, 1.2, num_vals),
}

# Get parameter names and values
keys = list(params.keys())
values = list(params.values())

# Generate all combinations and ensure values are Python floats
meshgrid_dicts = [dict(zip(keys, map(float, combination))) for combination in itertools.product(*values)]

# Print results
# for d in meshgrid_dicts:
#     print(d)
j = 1
for param in np.random.choice(meshgrid_dicts, size=min(10, len(meshgrid_dicts)), replace=False):
    #params_to_try = params.copy()
    #params_to_try.update(param)
    print('PARAM',j,param)
    j += 1
    model = HillTypeModelWrapper(param)


    bounds = torch.tensor([
        [MIN_STRETCHED_MUSCLE_LENGTH_ONE, MIN_STRETCHED_MUSCLE_LENGTH_TWO],
        [MAX_STRETCHED_MUSCLE_LENGTH_ONE, MAX_STRETCHED_MUSCLE_LENGTH_TWO]])
    initial_muscle_lengths = draw_sobol_samples(
        bounds=bounds, n=1, q=NUM_INITIAL_POINTS).squeeze(0).to(torch.double)
    range_of_motions = model.simulate_forward_for_botorch(initial_muscle_lengths)


    for iteration in range(NUM_ITERATIONS-1):
        # min-max scaling
        initial_muscle_lengths = (initial_muscle_lengths - bounds[0][0]) / (bounds[1][0] - bounds[0][0])
        range_of_motions = (range_of_motions - bounds[0][1]) / (bounds[1][1] - bounds[0][1])

        # print('initial_muscle_lengths',initial_muscle_lengths)
        # print('range_of_motions',range_of_motions)

        gp = gp_process(initial_muscle_lengths, range_of_motions)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        acqf = acq_func(gp)

        candidate_muscle_length, _ = optimize_acqf(
            acq_function=acqf, bounds=bounds, q=NUM_NEW_CANDIDATES,
            num_restarts=50, raw_samples=200)

        new_range_of_motion = model.simulate_forward_for_botorch(
            candidate_muscle_length)
        
        # Problem: new_range_of_motion contains infinity
        # Solution: set infinity to 0
        if torch.isinf(new_range_of_motion).sum() > 0:
            warnings.warn("Range of motion contains infinity: {}".format(new_range_of_motion))

        new_range_of_motion[torch.isinf(new_range_of_motion)] = 0
       
        initial_muscle_lengths = torch.cat([initial_muscle_lengths,
                                            candidate_muscle_length])
        range_of_motions = torch.cat([range_of_motions, new_range_of_motion])

    initial_muscle_lengths = initial_muscle_lengths.numpy()
    range_of_motions = range_of_motions.numpy()

    plotter = RangeOfMotionPlotter(initial_muscle_lengths, range_of_motions)
    plotter.save_as_csv()
    # plotter.plot()
