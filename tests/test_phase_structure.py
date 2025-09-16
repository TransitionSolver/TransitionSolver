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
from dictcmp import isclose

THIS = Path(os.path.dirname(os.path.abspath(__file__)))


def test_phase_structure():
    point = THIS / 'rss_bp1.txt'
    phase_structure_file = THIS / "rss_bp1_phase_structure.dat"
    potential = RealScalarSingletModel(*np.loadtxt(point))
    result = phasetracer.phase_structure(
        potential, phase_structure_file, vw=0.9)
    # TODO ignoring T and SonT as the size of them changes between local & github
    # but good to reinclude them later
    assert isclose(result, THIS / "rss_bp1_phase_structure.json", ignore=['analysisTime', 'T', 'SonT'], rtol=5e-2, atol=1e-9)
