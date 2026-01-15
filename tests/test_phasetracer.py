"""
Check PhaseTracer installation
==============================
"""

import os
import subprocess
from pathlib import Path

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
    exe_name = build_phase_tracer("RSS.hpp", force=True)
    assert exe_name == phasetracer.PT_HOME / "RSS"


def test_run_phase_tracer(generate_baseline):
    exe_name = build_phase_tracer("RSS.hpp", force=True)
    phase_structure_file = BASELINE / "rss_bp1_phase_structure.dat"

    result = run_phase_tracer(exe_name, point=benchmarks.RSS_BP1_POINT)

    if generate_baseline:
        with open(phase_structure_file, 'w') as f:
            f.write(result)

    with open(phase_structure_file) as f:
        phase_structure_raw = f.read()

    assert result == phase_structure_raw
