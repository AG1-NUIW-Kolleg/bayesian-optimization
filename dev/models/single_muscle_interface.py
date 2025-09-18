from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import torch


class SingleMuscleInterface(ABC):
    @abstractmethod
    def simulate_forward_step(self, prestretch_force: float) -> float:
        pass

    def simulate_forward_for_botorch(self, prestretch_force):
        prestretch = prestretch_force.detach().numpy()
        range_of_motion = self.simulate_forward_step(prestretch)
        range_of_motion = torch.tensor([[range_of_motion]]).to(torch.double)
        return range_of_motion
