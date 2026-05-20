"""
Test access to CosmoTransitions methods
=======================================
"""


from TransitionSolver import RSS_BP2 as model


def test_cosmo():
    model.getPhases()
    model.findAllTransitions()
