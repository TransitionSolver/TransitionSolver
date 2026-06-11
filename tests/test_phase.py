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


THIS = Path(os.path.dirname(os.path.abspath(__file__)))
BASELINE = THIS / "baseline"


def test_find_phase_at_t():
    phase_structure = read_phase_tracer(
        phase_tracer_file=BASELINE / "rss_bp1_phase_structure.dat"
    )
    phase = phase_structure.phases[0]

    result = phase.find_phase_at_t(200., benchmarks.RSS_BP1)

    assert np.allclose(result, np.array([2.85775445e-04, 6.47017016e+02]))

