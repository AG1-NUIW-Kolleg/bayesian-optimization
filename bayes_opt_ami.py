from __future__ import annotations

import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.optim.stopping import ExpMAStoppingCriterion
from botorch.utils import draw_sobol_samples
from gpytorch import ExactMarginalLogLikelihood

from dev.constants import ADDITIONAL_STRETCH_LENGTH
from dev.constants import NUM_INITIAL_POINTS
from dev.constants import NUM_NEW_CANDIDATES
from dev.constants import SEED
from dev.models.hill_type_model_wrapper import HillTypeModelWrapper
from dev.visual.range_of_motion_plotter import RangeOfMotionPlotter


def acq_func(gaussian_process):
    return UpperConfidenceBound(gaussian_process, beta=0.1)


def gp_process(x, y):
    return SingleTaskGP(x, y)


relaxed_muscle_lengths = [
    (9.0, 12.0),
    (11.0, 12.0),
    (11.0, 15.0),
    (12.0, 13.0),
    (14.0, 15.0)
]

torch.manual_seed(SEED)

for length_pair in relaxed_muscle_lengths:

    params = {
        'Length_Slack_M1': length_pair[0],
        'Length_Slack_M2': length_pair[1],
    }

    model = HillTypeModelWrapper(params)

    max_stretched_muscle_length_one = \
        length_pair[0] + ADDITIONAL_STRETCH_LENGTH
    max_stretched_muscle_length_two = \
        length_pair[1] + ADDITIONAL_STRETCH_LENGTH

    bounds = torch.tensor([
        [length_pair[0], length_pair[1]],
        [max_stretched_muscle_length_one, max_stretched_muscle_length_two]])

    initial_muscle_lengths = draw_sobol_samples(
        bounds=bounds, n=1, q=NUM_INITIAL_POINTS).squeeze(0).to(torch.double)
    range_of_motions = model.simulate_forward_for_botorch(
        initial_muscle_lengths)

    stopper = ExpMAStoppingCriterion(n_window=2, minimize=False)
    is_optimization_converged = False

    while (is_optimization_converged is False):
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

        is_optimization_converged = stopper.evaluate(new_range_of_motion)

    initial_muscle_lengths = initial_muscle_lengths.numpy()
    range_of_motions = range_of_motions.numpy()

    plotter = RangeOfMotionPlotter(
        initial_muscle_lengths, range_of_motions, params)
    plotter.save_as_csv(f'm1_{length_pair[0]}_m2_{length_pair[1]}')
    plotter.plot()
