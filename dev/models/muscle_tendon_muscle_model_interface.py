from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class MuscleTendonMuscleModelInterface(ABC):
    @abstractmethod
    def simulate_forward_step(self):
        pass
