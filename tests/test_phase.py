"""
Tests of phase helpers
Specifically find_at_phase_t which calls scipy.optimize.fmin_powell
===================================================================
"""

import os
from pathlib import Path

import numpy as np
import pytest

from TransitionSolver import benchmarks, read_phase_tracer
from TransitionSolver.analysis.phase_structure import Phase, PhaseStructure


THIS = Path(os.path.dirname(os.path.abspath(__file__)))
BASELINE = THIS / "baseline"


def test_find_phase_at_t():
    phase_structure = read_phase_tracer(
        phase_tracer_file=BASELINE / "rss_bp1_phase_structure.dat"
    )
    phase = phase_structure.phases[0]

    result = phase.find_phase_at_t(200.0, benchmarks.RSS_BP1)

    assert np.allclose(result, np.array([2.85775445e-04, 6.47017016e02]))


def test_ground_state_energy_density():
    phases = [
        Phase(0, np.array([[0.0, 5.0, 0.0], [1.0, 4.0, 0.0]])),
        Phase(1, np.array([[0.0, -2.0, 1.0], [1.0, -1.0, 1.0]])),
    ]
    assert PhaseStructure(phases=phases).ground_state_energy_density == -2.0


def test_missing_ground_state_uses_radiation_domination():
    phase = Phase(0, np.array([[0.1, 5.0, 0.0], [1.0, 4.0, 0.0]]))
    phase_structure = PhaseStructure(phases=[phase])

    with pytest.warns(RuntimeWarning, match="Assuming radiation domination"):
        assert phase_structure.ground_state_energy_density is None
