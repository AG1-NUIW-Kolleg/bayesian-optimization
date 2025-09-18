from __future__ import annotations

import os
import subprocess


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
        env['PATH'] = \
            '/usr/local/home/cmcs-fa01/opendihu-elise/dependencies/python/install/bin:' + \
            env['PATH']
        env['PWD'] = self._simulation_script_path + '/build_release'

        subprocess.run(
            './muscle_with_prestretch ../settings_muscle_with_prestretch.py --force 1.1',
            shell=True, check=True, env=env, cwd=env['PWD'])

        range_of_motion = self._parser.parse_range_of_motion()

        return range_of_motion
