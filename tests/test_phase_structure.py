""""
Test phase structure report
===========================
"""

import os
from pathlib import Path

import pytest

from TransitionSolver import phasehistory, benchmarks, read_phase_tracer
from dictcmp import assert_deep_equal

THIS = Path(os.path.dirname(os.path.abspath(__file__)))
BASELINE = THIS / "baseline"


@pytest.mark.parametrize("name", ["RSS_BP1"])
def test_phase_structure(generate_baseline, name):
    phase_tracer_file = BASELINE / f"{name.lower()}_phase_structure.dat"
    with open(phase_tracer_file) as f:
        phase_tracer_data = f.read()
    phase_structure = read_phase_tracer(phase_tracer_data)
    model = getattr(benchmarks, name)
    result = phasehistory.find_phase_history(
        model, phase_structure, bubble_wall_velocity=1)
    assert_deep_equal(result, BASELINE / f"{name.lower()}_phase_structure.json", exclude_types=[list], significant_digits=4, generate_baseline=generate_baseline)
