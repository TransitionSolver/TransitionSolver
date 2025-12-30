"""
Test access to CosmoTransitions methods
=======================================
"""


from TransitionSolver import RSS_BP1


def test_cosmo():
    RSS_BP1.getPhases()
    RSS_BP1.findAllTransitions()

