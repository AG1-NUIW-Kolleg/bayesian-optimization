from __future__ import annotations

from dev.models.hill_type_model_wrapper import HillTypeModelWrapper
params = {
    'Length_Slack_M1': 13,
    'Length_Slack_M2': 13,
}
model = HillTypeModelWrapper(params)

range_of_motion = model.simulate_forward_step(13.0, 13.0)
print(range_of_motion)
