""""
Test phase structure report
===========================
"""

import json
import os
from pathlib import Path

import numpy as np

from TransitionSolver import phasetracer
from TransitionSolver.models.real_scalar_singlet_model import RealScalarSingletModel

THIS = Path(os.path.dirname(os.path.abspath(__file__)))


def test_phase_structure():
    point = THIS / 'rss_bp1.txt'
    phase_structure_file = THIS / "rss_bp1_phase_structure.dat"
    potential = RealScalarSingletModel(*np.loadtxt(point))
    result = phasetracer.phase_structure(
        potential, phase_structure_file, vw=0.9)

    # sanitise result to avoid float/np.float64 differences
    # with disk
    result = json.loads(json.dumps(result))

    with open(THIS / "rss_bp1_phase_structure.json", "r") as f:
        expected = json.load(f)

    del result['analysisTime']
    del expected['analysisTime']

    assert result == expected
