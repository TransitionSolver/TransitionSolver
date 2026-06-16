"""
Test model, including access to CosmoTransitions methods
========================================================
"""

import numpy as np
import pytest

from TransitionSolver import benchmarks, RSS_BP2 as model


@pytest.mark.skip(reason="we found this to be platform dependent and recommend PT by default")
def test_cosmo():
    model.getPhases()
    model.findAllTransitions()


POTENTIALS = [
    ("RSS_BP1", 6807486471.394539),
    ("RSS_BP2", 3869184455.127249),
    ("RSS_BP3", 27524540814.135723),
    ("RSS_BP4", 5199730932.906029),
    ("RSS_BP13", 3526627688.129584),
]


@pytest.mark.parametrize("name, expected", POTENTIALS)
def test_potential(name, expected):
    test_model = getattr(benchmarks, name)
    assert np.isclose(test_model(np.array([10., -10.]), 20.), expected)
