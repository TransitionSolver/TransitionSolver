"""
Test action calculations
========================
"""

import os
from pathlib import Path

import numpy as np

from TransitionSolver import action, benchmarks, read_phase_tracer


THIS = Path(os.path.dirname(os.path.abspath(__file__)))
BASELINE = THIS / "baseline"


def test_action_ct():
    phase_structure = read_phase_tracer(
        phase_tracer_file=BASELINE / "rss_bp1_phase_structure.dat"
    )
    false_vacuum = phase_structure.phases[0].find_phase_at_t(190., benchmarks.RSS_BP1)
    true_vacuum = phase_structure.phases[1].find_phase_at_t(190., benchmarks.RSS_BP1)

    result = action.action_ct(benchmarks.RSS_BP1, 190., false_vacuum, true_vacuum)

    assert np.isclose(result, 2.1768231378296363)


def test_action_pt():
    phase_structure = read_phase_tracer(
        phase_tracer_file=BASELINE / "rss_bp1_phase_structure.dat"
    )
    false_vacuum = phase_structure.phases[0].find_phase_at_t(190., benchmarks.RSS_BP1)
    true_vacuum = phase_structure.phases[1].find_phase_at_t(190., benchmarks.RSS_BP1)

    result = action.action_pt(benchmarks.RSS_BP1, 190., false_vacuum, true_vacuum)

    assert np.isclose(result, 2.151439938773795)
