from __future__ import annotations

import jax
import numpy as np

from dev.models.AD_Hill_System_HMC_Py import observe_blackbox_simulation
from dev.models.muscle_tendon_muscle_model_interface import \
    MuscleTendonMuscleModelInterface

from dev.models.AD_Hill_System_HMC_Py import extminobs_muscle_1
from dev.models.AD_Hill_System_HMC_Py import extmaxobs_muscle_1
from dev.models.AD_Hill_System_HMC_Py import extminobs_muscle_2
from dev.models.AD_Hill_System_HMC_Py import extmaxobs_muscle_2

class HillTypeModelWrapper(MuscleTendonMuscleModelInterface):
    def __init__(self, params=None):
        self._params = params

    def simulate_forward_step(self, stretched_muscle_length_one: float,
                              stretched_muscle_length_two: float) -> float:
        simulation_input = np.array(
            [stretched_muscle_length_one, stretched_muscle_length_two])

        data = observe_blackbox_simulation(simulation_input, self._params)
        data = jax.device_get(data)
        muscle_one_maximum_length = data[0]
        muscle_one_minimum_length = data[1]

        range_of_motion = muscle_one_maximum_length - muscle_one_minimum_length
        return range_of_motion

    def is_input_in_bounds(self, stretched_muscle_length_one: float,
                              stretched_muscle_length_two: float) -> bool:
        is_in_bounds = False

        if (stretched_muscle_length_one >= extminobs_muscle_1 and
            stretched_muscle_length_one <= extmaxobs_muscle_1 and
            stretched_muscle_length_two >= extminobs_muscle_2 and
            stretched_muscle_length_two <= extmaxobs_muscle_2):
            is_in_bounds = True

        return is_in_bounds