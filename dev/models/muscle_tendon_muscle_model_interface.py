from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import torch


class MuscleTendonMuscleModelInterface(ABC):
    @abstractmethod
    def simulate_forward_step(self, stretched_muscle_length_one: float,
                              stretched_muscle_length_two: float) -> float:
        pass

    def simulate_forward_for_botorch(self, stretched_muscle_lengths):
        stretched_muscle_lengths = stretched_muscle_lengths.numpy().squeeze()
        stretched_muscle_length_one = stretched_muscle_lengths[0]
        stretched_muscle_length_two = stretched_muscle_lengths[1]
        range_of_motion = self.simulate_forward_step(
            stretched_muscle_length_one, stretched_muscle_length_two)
        range_of_motion = torch.tensor([[range_of_motion]]).to(torch.double)
        return range_of_motion
