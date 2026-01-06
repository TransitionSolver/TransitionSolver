"""
Test various plots of phase transitions
=======================================
"""

import os
from pathlib import Path

import pytest

from TransitionSolver.plot import plot_summary


THIS = Path(os.path.dirname(os.path.abspath(__file__)))
BASELINE = THIS / "baseline"


@pytest.mark.mpl_image_compare
def test_summary():
    return plot_summary(phase_structure_file=BASELINE / "rss_bp1_phase_structure.json")

