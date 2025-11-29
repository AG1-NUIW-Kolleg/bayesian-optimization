from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pandas as pd

from dev.constants import FILEPATH_OUTPUT

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


class RangeOfMotionParser():
    def __init__(self, filepath=FILEPATH_OUTPUT):
        self._filepath = filepath + 'muscle_length_contraction.csv'
        self._iteration = 0

    def parse_range_of_motion(self):
        with open(self._filepath) as file:
            content = file.read()
            values = content.split(',')
            values = [float(x) for x in values if len(x) > 0]

        df = pd.DataFrame(values)
        min_length = df.min()[0]
        max_length = df.max()[0]
        range_of_motion = max_length - min_length

        self._iteration += 1
        self._copy_files_to_debug()

        return range_of_motion

    def _copy_files_to_debug(self):
        """Copy output files to debug folder for inspection."""
        project_root = Path(__file__).parent.parent.parent
        debug_dir = project_root / 'out' / 'debug' / f'{self._iteration}'
        debug_dir.mkdir(parents=True, exist_ok=True)

        print(f"Debug: Attempting to copy files to {debug_dir}")
        print(f"Debug: Source directory: {Path(self._filepath).parent}")

        source_dir = Path(self._filepath).parent

        # Files with their relative paths from source_dir
        files_to_copy = {
            'log.csv': source_dir / 'logs' / 'log.csv',
            'muscle_length_contraction.csv': source_dir / 'muscle_length_contraction.csv',
            'muscle_length_prestretch.csv': source_dir / 'muscle_length_prestretch.csv',
            'solver_structure.txt': source_dir / 'solver_structure.txt'
        }

        for dest_filename, source in files_to_copy.items():
            if source.exists():
                dest = debug_dir / dest_filename
                shutil.copy2(source, dest)
                print(f"Debug: Copied {dest_filename}")
            else:
                print(f"Debug: File not found: {source}")
