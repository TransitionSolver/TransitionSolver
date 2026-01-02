"""
Compute action by CosmoTransitions or PhaseTracer
=================================================
"""

import sys
import os
import warnings
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np
import cppyy
from cosmoTransitions import pathDeformation

from . import eigen


PT_HOME = Path(os.getenv("PHASETRACER", Path.home() / ".PhaseTracer"))
PT_LIB = PT_HOME / "lib" / "libphasetracer.so"
PT_INCLUDE = PT_HOME / "include"


cppyy.load_library(str(PT_LIB))
cppyy.add_include_path(str(PT_INCLUDE))
cppyy.include(str(PT_INCLUDE / 'action_calculator.hpp'))
cppyy.cppdef('LOGGER(error)')


from cppyy.gbl import PhaseTracer


def action_ct(potential, T, false_vacuum, true_vacuum, verbose=False, **kwargs):
    """
    @returns Action from CosmoTransitions's path deformation algorithm
    """
    with redirect_stdout(sys.stdout if verbose else None):
        return pathDeformation.fullTunneling([true_vacuum, false_vacuum], lambda X: potential(X, T), lambda X: potential.grad(X, T),
            verbose=verbose, **kwargs).action
            

def action_pt(potential, T, false_vacuum, true_vacuum, **kwargs):
    """
    @returns Action from PhaseTracer's path deformation algorithm
    """
    action_calculator = PhaseTracer.ActionCalculator(potential)
    action_calculator.__python_owns__ = False
    action = action_calculator.get_action(eigen.vector(true_vacuum), eigen.vector(false_vacuum), T)
    if np.isnan(action) or action < 0:
        warnings.warn(f"Failed to compute action at T = {T}. action = {action}")
        return None
    return action
    

