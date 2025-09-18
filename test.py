from __future__ import annotations

from dev.constants import FILEPATH_OUTPUT
from dev.models.dummy_cuboid_wrapper import DummyCuboidWrapper
from dev.util.range_of_motion_parser import RangeOfMotionParser

script_path = \
    '/usr/local/home/cmcs-fa01/opendihu-elise/examples/electrophysiology/neuromuscular/cuboid_muscle_with_prestretch_4x4/'

parser = RangeOfMotionParser(FILEPATH_OUTPUT)

model = DummyCuboidWrapper(script_path, parser)
range_of_motion = model.simulate_dummy_step()
print(range_of_motion)
