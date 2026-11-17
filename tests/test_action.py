"""
Test action calculations
========================
"""

import os
from pathlib import Path

import numpy as np

from TransitionSolver import action, read_phase_tracer
from TransitionSolver.analysis.transition_analysis import PD_DEFAULT_SETTINGS
from TransitionSolver.benchmarks import RSS_BP1

THIS = Path(os.path.dirname(os.path.abspath(__file__)))
BASELINE = THIS / "baseline"

FALSE_VACUUM_AT_190 = np.array([2.7185364877499842e-05, 645.4717470219503])
TRUE_VACUUM_AT_190 = np.array([1.2354131737989441e-05, 632.050713823395])


def test_action_ct():
    phase_structure = read_phase_tracer(
        phase_tracer_file=BASELINE / "rss_bp1_phase_structure.dat"
    )
    false_vacuum = phase_structure.phases[0].find_phase_at_t(190.0, RSS_BP1)
    true_vacuum = phase_structure.phases[1].find_phase_at_t(190.0, RSS_BP1)

    result = action.action_ct(
        RSS_BP1, 190.0, false_vacuum, true_vacuum, **PD_DEFAULT_SETTINGS
    )

    assert np.isclose(result, 2.1760490153373664)


def test_action_pt():
    phase_structure = read_phase_tracer(
        phase_tracer_file=BASELINE / "rss_bp1_phase_structure.dat"
    )
    false_vacuum = phase_structure.phases[0].find_phase_at_t(190.0, RSS_BP1)
    true_vacuum = phase_structure.phases[1].find_phase_at_t(190.0, RSS_BP1)

    result = action.action_pt(
        RSS_BP1, 190.0, false_vacuum, true_vacuum, **PD_DEFAULT_SETTINGS
    )

    assert np.isclose(result, 2.1762490040416886)


def test_action_ct_fixed_minima():
    result = action.action_ct(
        RSS_BP1, 190.0, FALSE_VACUUM_AT_190, TRUE_VACUUM_AT_190, **PD_DEFAULT_SETTINGS
    )

    assert np.isclose(result, 2.1759622127052163)


def test_action_pt_fixed_minima():
    result = action.action_pt(
        RSS_BP1, 190.0, FALSE_VACUUM_AT_190, TRUE_VACUUM_AT_190, **PD_DEFAULT_SETTINGS
    )

    assert np.isclose(result, 2.176330531760892)
