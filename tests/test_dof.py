"""
Test the number of degrees of freedom
=====================================
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from TransitionSolver import phasetracer, RSS_BP1

from dictcmp import allclose

THIS = Path(os.path.dirname(os.path.abspath(__file__)))


def test_dof():
    phase_structure_file = THIS / "rss_bp1_phase_structure.dat"
    result = phasetracer.trace_dof(RSS_BP1, phase_structure_file)
    assert allclose(result, THIS / "rss_bp1_dof.json")


@pytest.mark.mpl_image_compare(tolerance=20)
def test_plot_dof():
    with open(THIS / "rss_bp1_dof.json", "r") as f:
        expected = json.load(f)

    fig, ax = plt.subplots()

    for k, v in expected.items():
        ax.plot(v["T"], v["dof"], label=f'Phase {k}')

    ax.invert_xaxis()
    ax.set_xscale('log')
    ax.set_xlabel('Temperature, $T$ (GeV)')
    ax.set_ylabel('Effective degrees of freedom')
    ax.legend()

    return fig
