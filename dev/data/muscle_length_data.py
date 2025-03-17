from __future__ import annotations


class MuscleLengthData():
    """Data class for storing muscle length"""

    def __init__(self):
        self._initial_length = 0.0  # in m
        self._minimum_length = 0.0  # in m
        self._maximum_length = 0.0  # in m

    @property
    def initial_length(self):
        return self._initial_length

    @property
    def minimum_length(self):
        return self._minimum_length

    @property
    def maximum_length(self):
        return self._maximum_length

    @initial_length.setter
    def initial_length(self, value: float):
        self._initial_length = value

    @minimum_length.setter
    def minimum_length(self, value: float):
        self._minimum_length = value

    @maximum_length.setter
    def maximum_length(self, value: float):
        self._maximum_length = value
