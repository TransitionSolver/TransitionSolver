"""
Test action calculations
========================
"""

import os
from pathlib import Path

import numpy as np

from TransitionSolver import action, read_phase_tracer
from TransitionSolver.benchmarks import RSS_BP1

THIS = Path(os.path.dirname(os.path.abspath(__file__)))
BASELINE = THIS / "baseline"


def test_action_ct():
    phase_structure = read_phase_tracer(
        phase_tracer_file=BASELINE / "rss_bp1_phase_structure.dat"
    )
    false_vacuum = phase_structure.phases[0].find_phase_at_t(190.0, RSS_BP1)
    true_vacuum = phase_structure.phases[1].find_phase_at_t(190.0, RSS_BP1)

    result = action.action_ct(RSS_BP1, 190.0, false_vacuum, true_vacuum)

    assert np.isclose(result, 2.188357753482806)


def test_action_pt():
    phase_structure = read_phase_tracer(
        phase_tracer_file=BASELINE / "rss_bp1_phase_structure.dat"
    )
    false_vacuum = phase_structure.phases[0].find_phase_at_t(190.0, RSS_BP1)
    true_vacuum = phase_structure.phases[1].find_phase_at_t(190.0, RSS_BP1)

    result = action.action_pt(RSS_BP1, 190.0, false_vacuum, true_vacuum)

    assert np.isclose(result, 2.177286191324302)
