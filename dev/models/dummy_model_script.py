from __future__ import annotations

import os
import sys

from dev.constants import FILEPATH_OUTPUT
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


with open(FILEPATH_OUTPUT, 'w') as file:
    file.write('187')
