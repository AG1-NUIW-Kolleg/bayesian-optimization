from __future__ import annotations

import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils import draw_sobol_samples
from gpytorch import ExactMarginalLogLikelihood

from dev.constants import MAX_STRETCHED_MUSCLE_LENGTH_ONE
from dev.constants import MAX_STRETCHED_MUSCLE_LENGTH_TWO
from dev.constants import MIN_STRETCHED_MUSCLE_LENGTH_ONE
from dev.constants import MIN_STRETCHED_MUSCLE_LENGTH_TWO
from dev.constants import NUM_INITIAL_POINTS
from dev.constants import NUM_ITERATIONS
from dev.constants import NUM_NEW_CANDIDATES
from dev.constants import SEED
from dev.models.hill_type_model_wrapper import HillTypeModelWrapper
from dev.visual.range_of_motion_plotter import RangeOfMotionPlotter

torch.manual_seed(SEED)


def acq_func(gaussian_process):
    return UpperConfidenceBound(gaussian_process, beta=0.1)


def gp_process(x, y):
    return SingleTaskGP(x, y)


params = {
    'Length_Slack_M1': 9,
    'Length_Slack_M2': 12,
}
model = HillTypeModelWrapper(params)

bounds = torch.tensor([
    [MIN_STRETCHED_MUSCLE_LENGTH_ONE, MIN_STRETCHED_MUSCLE_LENGTH_TWO],
    [MAX_STRETCHED_MUSCLE_LENGTH_ONE, MAX_STRETCHED_MUSCLE_LENGTH_TWO]])
initial_muscle_lengths = draw_sobol_samples(
    bounds=bounds, n=1, q=NUM_INITIAL_POINTS).squeeze(0).to(torch.double)
range_of_motions = model.simulate_forward_for_botorch(initial_muscle_lengths)


for iteration in range(NUM_ITERATIONS-1):
    gp = gp_process(initial_muscle_lengths, range_of_motions)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    acqf = acq_func(gp)

    candidate_muscle_length, _ = optimize_acqf(
        acq_function=acqf, bounds=bounds, q=NUM_NEW_CANDIDATES,
        num_restarts=50, raw_samples=200)

    new_range_of_motion = model.simulate_forward_for_botorch(
        candidate_muscle_length)
    initial_muscle_lengths = torch.cat([initial_muscle_lengths,
                                        candidate_muscle_length])
    range_of_motions = torch.cat([range_of_motions, new_range_of_motion])

initial_muscle_lengths = initial_muscle_lengths.numpy()
range_of_motions = range_of_motions.numpy()

plotter = RangeOfMotionPlotter(initial_muscle_lengths, range_of_motions)
plotter.save_as_csv()
plotter.plot()
