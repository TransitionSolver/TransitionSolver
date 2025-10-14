"""
PhaseTracer interface
=====================
"""

import os
from pathlib import Path

import cppyy


PT_HOME = Path(os.getenv("PHASETRACER", Path.home() / ".PhaseTracer"))
PT_LIB = PT_HOME / "lib" / "libphasetracer.so"
PT_INCLUDE = PT_HOME / "include"


cppyy.load_library(str(PT_LIB))
cppyy.add_include_path(str(PT_INCLUDE))
cppyy.include(str(PT_INCLUDE / 'phase_finder.hpp'))
cppyy.include(str(PT_INCLUDE / 'transition_finder.hpp'))

from cppyy.gbl import PhaseTracer


def run_phase_tracer(potential):
    pf = PhaseTracer.PhaseFinder(potential)
    phases = pf.find_phases();
    tf = PhaseTracer.TransitionFinder(pf)
    transitions = tf.find_transitions();
    return phases, transitions

