from __future__ import annotations

from dev.models.hill_type_model_wrapper import HillTypeModelWrapper


def test_does_forward_simulation_return_float():
    model = HillTypeModelWrapper()
    stretched_muscle_length = 15.0

    result = model.simulate_forward_step(
        stretched_muscle_length, stretched_muscle_length)

    assert isinstance(result, float)


def test_does_too_small_input_return_false():
    model = HillTypeModelWrapper()
    too_small_length = 5.0

    is_in_bounds = model.is_input_in_bounds(too_small_length, too_small_length)

    assert (is_in_bounds is False)


def test_does_too_large_input_return_false():
    model = HillTypeModelWrapper()
    too_large_lenght = 23.0

    is_in_bounds = model.is_input_in_bounds(too_large_lenght, too_large_lenght)

    assert (is_in_bounds is False)


def test_does_input_in_bounds_return_true():
    model = HillTypeModelWrapper()
    fitting_length = 14.0

    is_in_bounds = model.is_input_in_bounds(fitting_length, fitting_length)

    assert (is_in_bounds is True)
