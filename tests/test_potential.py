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
    assert np.isclose(model([10., -10.], 20.), 100)
