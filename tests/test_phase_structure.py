""""
Test phase structure report
===========================
"""

import json
import os
from pathlib import Path

import numpy as np

from TransitionSolver.analysis.phase_structure import load_data
from TransitionSolver import phasehistory, RSS_BP1, read_phase_tracer
from dictcmp import allclose

THIS = Path(os.path.dirname(os.path.abspath(__file__)))


def test_phase_structure_by_file():
    phase_structure_file = THIS / "rss_bp1_phase_structure.dat"
    result = phasehistory.find_phase_history(
        RSS_BP1, phase_structure_file=phase_structure_file, vw=0.9)
    # TODO ignoring T and SonT as the size of them changes between local & github
    # but good to reinclude them later
    assert allclose(result, THIS / "rss_bp1_phase_structure.json", ignore=['analysisTime', 'T', 'SonT'], rtol=5e-2, atol=1e-9)


def test_phase_structure_by_object():
    phase_structure_file = THIS / "rss_bp1_phase_structure.dat"
    with open(phase_structure_file) as f:
        phase_structure_raw = f.read()
    phase_structure = read_phase_tracer(phase_structure_raw)
    result = phasehistory.find_phase_history(
        RSS_BP1, phase_structure, vw=0.9)
    # TODO ignoring T and SonT as the size of them changes between local & github
    # but good to reinclude them later
    assert allclose(result, THIS / "rss_bp1_phase_structure.json", ignore=['analysisTime', 'T', 'SonT'], rtol=5e-2, atol=1e-9)


def test_read_phase_tracer():
    phase_structure_file = THIS / "rss_bp1_phase_structure.dat"
    with open(phase_structure_file) as f:
        phase_structure_raw = f.read()
    result = read_phase_tracer(phase_structure_raw)
    expected = load_data(phase_structure_file)[1]
    for a, b in zip(result.phases, expected.phases):
        assert (a.raw == b.raw).all()
    for a, b in zip(result.transitions, expected.transitions):
        assert (a.raw == b.raw).all()
    assert result.transitionPaths == expected.transitionPaths
