"""
Test various plots of phase transitions
=======================================
"""

import os
from pathlib import Path

import pytest

from TransitionSolver.phasetracer import plot_action_curve

THIS = Path(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.mpl_image_compare
def test_action_curve():
    phase_structure_file = THIS / "rss_bp1_phase_structure.json"
    transition_id = 0
    ax = plot_action_curve(transition_id, phase_structure_file)
    return ax.get_figure()
