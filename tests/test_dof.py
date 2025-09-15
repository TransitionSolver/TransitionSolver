"""
Test the number of degrees of freedom
=====================================
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from TransitionSolver import phasetracer
from TransitionSolver.models.real_scalar_singlet_model import RealScalarSingletModel

THIS = Path(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.mpl_image_compare
def test_dof():
    point = THIS / 'rss_bp1.txt'
    phase_structure_file = THIS / "rss_bp1_phase_structure.dat"
    potential = RealScalarSingletModel(*np.loadtxt(point))
    data = phasetracer.trace_dof(potential, phase_structure_file)

    fig, ax = plt.subplots()

    for key, (T, dof) in data.items():
        ax.plot(T, dof, linewidth=2.5, label=f'Phase {key}')

    ax.invert_xaxis()
    ax.set_xscale('log')
    ax.set_xlabel('Temperature, $T$ (GeV)')
    ax.set_ylabel('Effective degrees of freedom')
    ax.legend()

    return fig
