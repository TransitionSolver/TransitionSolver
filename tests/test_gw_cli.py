"""Test gravitational-wave post-processing helpers."""

import pytest

from TransitionSolver.cli import (
    transition_diagnostics,
    transitions_with_percolation_temperature,
    valid_transition_ids,
)


def test_valid_transition_ids_are_unique():
    report = {
        "paths": [
            {"valid": True, "transitions": ["0", "1"]},
            {"valid": True, "transitions": ["1"]},
            {"valid": False, "transitions": ["2"]},
        ]
    }

    assert valid_transition_ids(report) == ["0", "1"]


def test_transitions_with_percolation_temperature_include_invalid_paths():
    report = {
        "paths": [{"valid": True, "transitions": ["0"]}],
        "transitions": {
            "0": {"T_p": 100.0},
            "1": {"T_p": 50.0},
            "2": {"T_p": None},
        },
    }

    assert transitions_with_percolation_temperature(report) == ["0", "1"]


@pytest.mark.parametrize(
    "decreasing_temperature, expected_warning",
    [
        (
            50.0,
            "does, however, begin decreasing later",
        ),
        (
            None,
            "Global physical percolation and completion are therefore not "
            "established",
        ),
    ],
)
def test_physical_volume_warning(decreasing_temperature, expected_warning):
    diagnostic = transition_diagnostics(
        {
            "T_f": None,
            "decreasing_v_phys_p": False,
            "T_decreasing_v_phys": decreasing_temperature,
        }
    )

    assert not diagnostic["Completion temperature exists"]
    assert not diagnostic["Physical false-vacuum volume decreasing at T_p"]
    assert diagnostic[
        "Temperature where physical false-vacuum volume begins decreasing"
    ] == decreasing_temperature
    assert any(expected_warning in warning for warning in diagnostic["Warnings"])
