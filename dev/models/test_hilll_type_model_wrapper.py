from __future__ import annotations

from dev.models.hill_type_model_wrapper import HillTypeModelWrapper


def test_does_forward_simulation_return_float():
    model = HillTypeModelWrapper()
    stretched_muscle_length = 15.0

    result = model.simulate_forward_step(
        stretched_muscle_length, stretched_muscle_length)

    assert isinstance(result, float)
