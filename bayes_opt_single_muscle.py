from __future__ import annotations

import random

import pandas as pd
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
    '/usr/local/home/cmcs-fa01/opendihu-elise/examples/electrophysiology/neuromuscular/cuboid_4x4_prestretch_general/'

parser = RangeOfMotionParser(FILEPATH_OUTPUT)

model = CuboidWrapper(script_path, parser)

bounds = torch.tensor([[0.0], [ADDITIONAL_STRETCH_FORCE]])

initial_prestretch_force = random.uniform(0, ADDITIONAL_STRETCH_FORCE)
initial_prestretch_force = torch.tensor(
    [[initial_prestretch_force],], dtype=torch.double)

range_of_motions = model.simulate_forward_for_botorch(initial_prestretch_force)
X_plot = torch.linspace(
    bounds[0].item(), bounds[1].item(), steps=400, dtype=torch.double
).unsqueeze(-1)

stopper = ExpMAStoppingCriterion(n_window=2, minimize=False)
is_optimization_converged = False
iteration = 0

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

    is_optimization_converged = stopper.evaluate(new_range_of_motion)

    was_training = gp.training
    was_lik_training = gp.likelihood.training

    gp.eval()
    gp.likelihood.eval()

    with torch.no_grad():
        post = gp.posterior(X_plot)
        mu = post.mean.squeeze(-1)
        sigma = post.variance.squeeze(-1).clamp_min(0).sqrt()

    snapshot = {
        "iter": iteration,

        # data the GP was fit on THIS iteration (before appending the new point)
        "train_X": initial_prestretch_force.detach().cpu(),
        "train_Y": range_of_motions.detach().cpu(),

        # fitted GP parameters
        "model_state": {k: v.detach().cpu() for k, v in gp.state_dict().items()},
        "likelihood_state": {k: v.detach().cpu() for k, v in gp.likelihood.state_dict().items()},

        # optional, but makes plotting trivial/reproducible
        "X_plot": X_plot.detach().cpu(),
        "mu": mu.detach().cpu(),
        "sigma": sigma.detach().cpu(),

        # what BO picked/observed this iteration
        "candidate_X": candidate_prestretch_force.detach().cpu(),
        "candidate_Y": new_range_of_motion.detach().cpu(),

        #stopper convergence criterion
        "is_optimization_converged": is_optimization_converged,

    }
    torch.save(snapshot, f"out/bo_cuboid_gp_iter_{iteration:04d}.pt")

    if was_training:
        gp.train()
    if was_lik_training:
        gp.likelihood.train()

    initial_prestretch_force = torch.cat([initial_prestretch_force,
                                          candidate_prestretch_force])
    range_of_motions = torch.cat([range_of_motions, new_range_of_motion])

    iteration += 1
    print(f'Iteration {iteration} finished.')

initial_prestretch_force = initial_prestretch_force.numpy()
range_of_motions = range_of_motions.numpy()

df = pd.DataFrame({
    'prestretch_force': [tensor.item() for tensor in initial_prestretch_force],
    'range_of_motion': [tensor.item() for tensor in range_of_motions],
})
df.to_csv('out/bo_cuboid.csv')
