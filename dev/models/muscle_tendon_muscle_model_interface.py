from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import torch


class MuscleTendonMuscleModelInterface(ABC):
    @abstractmethod
    def simulate_forward_step(self, muscle_length_one: float,
                              muscle_length_two: float) -> float:
        pass

    def simulate_forward_for_botorch(self, muscle_lengths):
        muscle_lengths = muscle_lengths.numpy().squeeze()
        muscle_length_one = muscle_lengths[0]
        muscle_length_two = muscle_lengths[1]
        range_of_motion = self.simulate_forward_step(muscle_length_one,
                                                     muscle_length_two)
        range_of_motion = torch.tensor([[range_of_motion]]).to(torch.double)
        return range_of_motion
