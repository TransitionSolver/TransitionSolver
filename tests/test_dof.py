"""
Test the number of degrees of freedom
=====================================
"""

import os
from pathlib import Path

import json_numpy as json
import matplotlib.pyplot as plt
import pytest

from TransitionSolver import phasehistory, RSS_BP1

from dictcmp import assert_deep_equal

THIS = Path(os.path.dirname(os.path.abspath(__file__)))
BASELINE = THIS / "baseline"


def test_dof(generate_baseline):
    phase_tracer_file = BASELINE / "rss_bp1_phase_structure.dat"
    result = phasehistory.trace_dof(RSS_BP1, phase_tracer_file=phase_tracer_file)
    assert_deep_equal(result, BASELINE / "rss_bp1_dof.json", generate_baseline=generate_baseline)


@pytest.mark.mpl_image_compare(tolerance=20)
def test_plot_dof():
    with open(THIS / "baseline" / "rss_bp1_dof.json", "r") as f:
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
