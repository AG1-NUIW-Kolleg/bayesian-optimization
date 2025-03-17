from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from dev.data.muscle_length_data import MuscleLengthData


class MuscleTendonMuscleModelInterface(ABC):
    @abstractmethod
    def simulate_forward_step(
            self, muscle_length_datas:
            tuple[MuscleLengthData, MuscleLengthData]) \
            -> tuple[MuscleLengthData, MuscleLengthData]:
        pass
