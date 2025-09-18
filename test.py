from __future__ import annotations

from dev.constants import FILEPATH_OUTPUT
from dev.models.cuboid_wrapper import CuboidWrapper
from dev.util.range_of_motion_parser import RangeOfMotionParser

script_path = \
    '/usr/local/home/cmcs-fa01/opendihu-elise/examples/electrophysiology/neuromuscular/cuboid_muscle_with_prestretch_4x4/'

parser = RangeOfMotionParser(FILEPATH_OUTPUT)

model = CuboidWrapper(script_path, parser)
range_of_motion = model.simulate_forward_step(1.1)
print(range_of_motion)
