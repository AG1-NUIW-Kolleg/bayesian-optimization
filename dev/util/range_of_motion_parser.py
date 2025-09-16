from __future__ import annotations

import os
import sys

from dev.constants import FILEPATH_OUTPUT
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


class RangeOfMotionParser():
    def __init__(self, filepath=FILEPATH_OUTPUT):
        self._filepath = filepath

    def parse_range_of_motion(self):
        range_of_motion = -1
        with open(self._filepath) as file:
            range_of_motion = file.read()

        return range_of_motion
