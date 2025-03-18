from __future__ import annotations

import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils import draw_sobol_samples
from gpytorch import ExactMarginalLogLikelihood

from dev.constants import MAX_LENGTH_MUSCLE_ONE
from dev.constants import MAX_LENGTH_MUSCLE_TWO
from dev.constants import MIN_LENGTH_MUSCLE_ONE
from dev.constants import MIN_LENGTH_MUSCLE_TWO
from dev.constants import SEED
from dev.models.hill_type_model_wrapper import HillTypeModelWrapper
from dev.visual.true_func_plot import plot_stretch_data_3d

torch.manual_seed(SEED)


def blackbox_model(x):
    model = HillTypeModelWrapper()
    x = x.numpy().squeeze()
    muscle_length_one = x[0]
    muscle_length_two = x[1]
    range_of_motion = model.simulate_forward_step(muscle_length_one,
                                                  muscle_length_two)
    range_of_motion = torch.tensor([[range_of_motion]]).to(torch.double)
    return range_of_motion


bounds = torch.tensor([[MIN_LENGTH_MUSCLE_ONE, MIN_LENGTH_MUSCLE_TWO],
                       [MAX_LENGTH_MUSCLE_ONE, MAX_LENGTH_MUSCLE_TWO]])

num_iterations = 10
initial_points = 1


def acq_func(gaussian_process): return UpperConfidenceBound(
    gaussian_process, beta=0.1)


def gp_process(x, y): return SingleTaskGP(x, y)


train_x = draw_sobol_samples(
    bounds=bounds, n=1, q=initial_points).squeeze(0).to(torch.double)
train_y = blackbox_model(train_x)
gp = gp_process(train_x, train_y)

for iteration in range(num_iterations-1):
    gp = gp_process(train_x, train_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    acqf = acq_func(gp)
    candidate, _ = optimize_acqf(
        acq_function=acqf, bounds=bounds, raw_samples=200, q=1,
        num_restarts=50)

    new_y = blackbox_model(candidate)
    train_x = torch.cat([train_x, candidate])
    train_y = torch.cat([train_y, new_y])

gp = gp_process(train_x, train_y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

train_x = train_x.numpy()
train_y = train_y.numpy()

data = []
for x, y in zip(train_x, train_y):
    stretch_dict = {
        'pre_stretches': x,
        'stretch_score': y[0],
    }
    data.append(stretch_dict)

plot_stretch_data_3d(data)
