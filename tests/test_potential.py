"""
Test model, including access to CosmoTransitions methods
========================================================
"""

import numpy as np
from TransitionSolver import RSS_BP2 as model


def test_cosmo():
    model.getPhases()
    model.findAllTransitions()

def test_potential():
    assert np.isclose(model(np.array([10., -10.]), 20.), 3869184455.127249)
