import torch
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from gpytorch import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound


from dev.constants import NUM_ITERATIONS
from dev.constants import NUM_NEW_CANDIDATES


from dev.visual.range_of_motion_plotter import RangeOfMotionPlotter

class DataAugmentation:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.bounds = args.bounds

    def acq_func(gaussian_process):
        return UpperConfidenceBound(gaussian_process, beta=0.1)

    def gp_process(x, y):
        return SingleTaskGP(x, y)

    def augment(self, image):
        for iteration in range(NUM_ITERATIONS-1):
            gp = self.gp_process(initial_muscle_lengths, range_of_motions)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            acqf = self.acq_func(gp)

            candidate_muscle_length, _ = optimize_acqf(
                acq_function=acqf, bounds=self.bounds, q=NUM_NEW_CANDIDATES,
                num_restarts=50, raw_samples=200)

            new_range_of_motion = self.model.simulate_forward_for_botorch(
                candidate_muscle_length)
            initial_muscle_lengths = torch.cat([initial_muscle_lengths,
                                                candidate_muscle_length])
            range_of_motions = torch.cat([range_of_motions, new_range_of_motion])

        initial_muscle_lengths = initial_muscle_lengths.numpy()
        range_of_motions = range_of_motions.numpy()

        # plotter = RangeOfMotionPlotter(initial_muscle_lengths, range_of_motions)
        # plotter.plot()
