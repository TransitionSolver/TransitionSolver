"""
Test various plots of phase transitions
=======================================
"""

import os
from pathlib import Path

import pytest

from TransitionSolver.plot import plot_action_curve, plot_vw

THIS = Path(os.path.dirname(os.path.abspath(__file__)))
phase_structure_file = THIS / "rss_bp1_phase_structure.json"
transition_id = 0

@pytest.mark.mpl_image_compare
def test_action_curve():
    ax = plot_action_curve(transition_id, phase_structure_file)
    return ax.get_figure()
    
@pytest.mark.mpl_image_compare
def test_vw():
    ax = plot_vw(transition_id, phase_structure_file)
    return ax.get_figure()
    
@pytest.mark.mpl_image_compare
def test_action_curve():
    ax = plot_action_curve(transition_id, phase_structure_file)
    return ax.get_figure()
