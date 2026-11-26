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
    assert np.isclose(kappa_nu_model(cs2, al, vp), 0.612825406284902)


def test_kappa_nu_model_ultrarelativistic_strong_transition():
    """Regression test for loss of ODE accuracy near the speed of light."""
    cs2 = 0.33281649099834998
    al = 1272.1657380174411
    vp = 0.999999

    kappa = kappa_nu_model(cs2, al, vp, use_cj=False)

    assert np.isclose(kappa, 0.9977817709087959, rtol=1e-7)
    assert 0 < kappa < 1
