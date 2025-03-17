import torch

from botorch.acquisition import UpperConfidenceBound
from botorch.models import SingleTaskGP
from gpytorch import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.utils import draw_sobol_samples
from models import HillType



def bo(blackbox_model = lambda x : HillType(x), acq_function = lambda gaussian_process: UpperConfidenceBound(gaussian_process, beta=0.1), gp_process = lambda x, y: SingleTaskGP(x, y),  bounds = torch.tensor([[12.,13.],[17.,18.]]), num_iterations = 10, initial_points = 1,):
    """
    Perform Bayesian Optimization on a given blackbox model.
    Parameters:
    -----------
    blackbox_model : callable
        The blackbox function to be optimized. It should take a tensor of input points and return a tensor of output values.
        Default: HillType() function from models.py.
    acq_function : callable, optional
        The acquisition function to be used for selecting the next point to evaluate. 
        Default: botorch.acquisition.UpperConfidenceBound with beta=0.1.
    gp_process : callable, optional
        The Gaussian Process model to be used. 
        Default: botorch.models.SingleTaskGP.
    bounds : torch.Tensor, optional
        A tensor specifying the bounds of the search space. 
        Default: torch.tensor([[12., 13.], [17., 18.]]).
    num_iterations : int, optional
        The number of iterations to perform. 
        Default: 10.
    initial_points : int, optional
        The number of initial points to sample using Sobol sampling.
        Default: 1.
    Returns:
    --------
    gp : botorch.models.SingleTaskGP
        The trained Gaussian Process model after performing Bayesian Optimization.
    train_x : torch.Tensor
        The tensor of input points evaluated during the optimization process.
    train_y : torch.Tensor
        The tensor of output values corresponding to the input points evaluated during the optimization process.
    """   
    
    train_x = draw_sobol_samples(bounds=bounds, n=1, q=initial_points).squeeze(0).to(torch.double)
    train_y = blackbox_model(train_x)
    gp = gp_process(train_x, train_y)

    for iteration in range(num_iterations-1):
        gp = gp_process(train_x, train_y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        acqf = acq_function(gp)
        candidate,_ = optimize_acqf(acq_function=acqf, bounds=bounds,raw_samples=200, q=1, num_restarts=50)

        new_y = blackbox_model(candidate)
        train_x=torch.cat([train_x, candidate])
        train_y = torch.cat([train_y, new_y])

    gp = gp_process(train_x, train_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp, train_x, train_y