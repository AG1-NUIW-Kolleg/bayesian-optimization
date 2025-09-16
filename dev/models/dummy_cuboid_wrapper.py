from __future__ import annotations

import subprocess
import sys


class DummyCuboidWrapper():
    def __init__(self, simulation_script_path, parser, params=None):
        self._simulation_script_path = simulation_script_path
        self._parser = parser
        self._params = params

    def simulate_dummy_step(self):
        # run dummy model script here
        subprocess.run([sys.executable, self._simulation_script_path],
                       check=True)

        range_of_motion = self._parser.parse_range_of_motion()

        return range_of_motion
