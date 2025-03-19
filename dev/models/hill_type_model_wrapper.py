from __future__ import annotations

import jax
import numpy as np

from dev.models.AD_Hill_System_HMC_Py import observe_blackbox_simulation
from dev.models.muscle_tendon_muscle_model_interface import \
    MuscleTendonMuscleModelInterface


class HillTypeModelWrapper(MuscleTendonMuscleModelInterface):
    def __init__(self, params):
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
