"""
Test kappa nu model
===================
"""

import numpy as np

from TransitionSolver.gws import kappa_nu_model


def test_kappa_nu_model():
    cs2 = 0.317458236000095
    al = 1.0219991793900685
    vp = 0.9288690647849615
    assert np.isclose(kappa_nu_model(cs2, al, vp), 0.6135879891493636)
