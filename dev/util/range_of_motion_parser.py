from __future__ import annotations

import os
import sys

import pandas as pd

from dev.constants import FILEPATH_OUTPUT

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


class RangeOfMotionParser():
    def __init__(self, filepath=FILEPATH_OUTPUT):
        self._filepath = filepath

    def parse_range_of_motion(self):
        with open(self._filepath) as file:
            content = file.read()
            values = content.split(',')
            values = [float(x) for x in values if len(x) > 0]

        df = pd.DataFrame(values)
        min_length = df.min()[0]
        max_length = df.max()[0]
        range_of_motion = max_length - min_length

        return range_of_motion
