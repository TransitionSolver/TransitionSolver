"""
Test gravitational waves
========================
"""

import json
import os
from pathlib import Path

import pytest
import numpy as np

from TransitionSolver.models.real_scalar_singlet_model import RealScalarSingletModel
from TransitionSolver.gws import lisa, GWAnalyser
from dictcmp import isclose


THIS = Path(os.path.dirname(os.path.abspath(__file__)))


point = THIS / 'rss_bp1.txt'
potential = RealScalarSingletModel(*np.loadtxt(point))
phase_structure_file = THIS / "rss_bp1_phase_structure.dat"
phase_history_file = THIS / "rss_bp1_phase_structure.json"

with open(THIS / phase_history_file, 'r') as f:
    phase_history = json.load(f)

analyser = GWAnalyser(potential, phase_structure_file, phase_history)


def test_report():
    report = analyser.report(detector=lisa)
    assert isclose(report, THIS / "rss_bp1_gw.json")


@pytest.mark.mpl_image_compare
def test_gw():
    return analyser.plot(detector=lisa)


def test_snr():
    snr = analyser.SNR(lisa)
    assert np.isclose(snr, 254.0820401210814)
