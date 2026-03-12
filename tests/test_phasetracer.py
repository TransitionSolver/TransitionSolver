"""
Check PhaseTracer installation
==============================
"""

import os
import subprocess
from pathlib import Path

import pytest


import TransitionSolver
from TransitionSolver import phasetracer, run_phase_tracer, build_phase_tracer, benchmarks


THIS = Path(os.path.dirname(os.path.abspath(__file__)))
BASELINE = THIS / "baseline"


def test_pt_home():
    assert phasetracer.PT_HOME.is_dir()


def test_pt_lib():
    assert phasetracer.PT_LIB.is_file()


def test_pt_unit_test():
    subprocess.check_call(phasetracer.PT_UNIT_TEST)


def test_build_phase_tracer():
    exe_name = build_phase_tracer("RSS_BP.hpp", force=True)
    assert exe_name == phasetracer.PT_HOME / "RSS_BP"

MODEL_NAME = [("RSS_BP", f"RSS_BP{k}") for k in range(1, 14)]

@pytest.mark.parametrize("model, name", MODEL_NAME)
def test_run_phase_tracer(generate_baseline, model, name):
    exe_name = build_phase_tracer("RSS_BP.hpp", force=True)
    phase_structure_file = BASELINE / f"{name.lower()}_phase_structure.dat"
    pt_settings_file = Path(TransitionSolver.__file__).resolve().parent / "pt_settings" / f"{model}.json"
    point = getattr(benchmarks, f"{name}_POINT")

    result = run_phase_tracer(exe_name, point=point, pt_settings_file=pt_settings_file)

    if generate_baseline:
        with open(phase_structure_file, 'w') as f:
            f.write(result)

    with open(phase_structure_file) as f:
        phase_structure_raw = f.read()

    assert result == phase_structure_raw
