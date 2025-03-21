from __future__ import annotations
from dev.models.hill_type_model_wrapper import HillTypeModelWrapper
from botorch.utils import draw_sobol_samples
from dev.constants import SEED
from dev.data_augmentation.data_augmentation import DataAugmentation
import torch
torch.manual_seed(SEED)

from dev.constants import MAX_LENGTH_MUSCLE_ONE
from dev.constants import MAX_LENGTH_MUSCLE_TWO
from dev.constants import MIN_LENGTH_MUSCLE_ONE
from dev.constants import MIN_LENGTH_MUSCLE_TWO
from dev.constants import NUM_INITIAL_POINTS


# define variables
# fixed variables: 
#   slack length of muscle one and two
#   maximum/ minimum extension of muscle one and two (for all ppl, don't change)
#   mass 
# random variables:
# TODO: define the random variables



model = HillTypeModelWrapper()

bounds = torch.tensor([[MIN_LENGTH_MUSCLE_ONE, MIN_LENGTH_MUSCLE_TWO],
                       [MAX_LENGTH_MUSCLE_ONE, MAX_LENGTH_MUSCLE_TWO]])
initial_muscle_lengths = draw_sobol_samples(
    bounds=bounds, n=1, q=NUM_INITIAL_POINTS).squeeze(0).to(torch.double)
range_of_motions = model.simulate_forward_for_botorch(initial_muscle_lengths)


data_aug = DataAugmentation(model, args={bounds: bounds})