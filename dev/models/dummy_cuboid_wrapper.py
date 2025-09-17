from __future__ import annotations

import subprocess
import sys
import os


class DummyCuboidWrapper():
    def __init__(self, simulation_script_path, parser, params=None):
        self._simulation_script_path = simulation_script_path
        self._parser = parser
        self._params = params

        self._project_root = \
            os.path.dirname(os.path.dirname(os.path.dirname(
                simulation_script_path)))

    def simulate_dummy_step(self):
        env = os.environ.copy()
        env['PYTHONPATH'] = self._project_root
        subprocess.run([sys.executable, self._simulation_script_path],
                       check=True, env=env)

        range_of_motion = self._parser.parse_range_of_motion()

        return range_of_motion
