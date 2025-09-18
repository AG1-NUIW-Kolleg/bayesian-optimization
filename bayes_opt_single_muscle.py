from __future__ import annotations

import random

import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.optim.stopping import ExpMAStoppingCriterion
from gpytorch import ExactMarginalLogLikelihood

from dev.constants import ADDITIONAL_STRETCH_FORCE
from dev.constants import FILEPATH_OUTPUT
from dev.constants import NUM_NEW_CANDIDATES
from dev.constants import SEED
from dev.models.cuboid_wrapper import CuboidWrapper
from dev.util.range_of_motion_parser import RangeOfMotionParser


def acq_func(gaussian_process):
    return UpperConfidenceBound(gaussian_process, beta=0.1)


def gp_process(x, y):
    return SingleTaskGP(x, y)


torch.manual_seed(SEED)

script_path = \
    '/usr/local/home/cmcs-fa01/opendihu-elise/examples/electrophysiology/neuromuscular/cuboid_muscle_with_prestretch_4x4/'

parser = RangeOfMotionParser(FILEPATH_OUTPUT)

model = CuboidWrapper(script_path, parser)

bounds = torch.tensor([[0, ADDITIONAL_STRETCH_FORCE]])

initial_prestretch_force = random.uniform(0, ADDITIONAL_STRETCH_FORCE)
initial_prestretch_force = torch.tensor(
    initial_prestretch_force, dtype=torch.double)

range_of_motions = model.simulate_forward_for_botorch(initial_prestretch_force)

stopper = ExpMAStoppingCriterion(n_window=2, minimize=False)
is_optimization_converged = False

while (is_optimization_converged is False):
    gp = gp_process(initial_prestretch_force, range_of_motions)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    acqf = acq_func(gp)

    candidate_prestretch_force, _ = optimize_acqf(
        acq_function=acqf, bounds=bounds, q=NUM_NEW_CANDIDATES,
        num_restarts=50, raw_samples=200)

    new_range_of_motion = model.simulate_forward_for_botorch(
        candidate_prestretch_force)

    initial_prestretch_force = torch.cat([initial_prestretch_force,
                                          candidate_prestretch_force])
    range_of_motions = torch.cat([range_of_motions, new_range_of_motion])

    is_optimization_converged = stopper.evaluate(new_range_of_motion)

initial_prestretch_force = initial_prestretch_force.numpy()
range_of_motions = range_of_motions.numpy()

# plotter = RangeOfMotionPlotter(
#     initial_prestretch_force, range_of_motions, params)
# plotter.save_as_csv(f'm1_{length_pair[0]}_m2_{length_pair[1]}')
# plotter.plot()
print(initial_prestretch_force)
print(range_of_motions)
