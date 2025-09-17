from __future__ import annotations

import os

from dev.constants import FILEPATH_OUTPUT
from dev.models.dummy_cuboid_wrapper import DummyCuboidWrapper
from dev.util.range_of_motion_parser import RangeOfMotionParser

script_path = os.path.join(
    os.path.dirname(__file__), 'dev', 'models', 'dummy_model_script.py')
parser = RangeOfMotionParser(FILEPATH_OUTPUT)

model = DummyCuboidWrapper(script_path, parser)
range_of_motion = model.simulate_dummy_step()
print(range_of_motion)
