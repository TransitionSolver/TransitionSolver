"""
Run PhaseTracer
===============
"""

import numpy as np

from . import geff
from .analysis.phase_history_analysis import PhaseHistoryAnalyser
from . import read_phase_tracer


def find_phase_history(potential, phase_structure=None, phase_tracer_file=None, bubble_wall_velocity=None, action_ct=True):  # TODO make false
    """
    @param potential Effective potential
    @param phase_structure Parsed phase structure from PT
    @param phase_tracer_file Results file from PT
    @param bubble_wall_velocity Bubble wall velocity

    @returns Report phase history from PhaseTracer output
    """
    if phase_tracer_file is not None:
        phase_structure = read_phase_tracer(phase_tracer_file=phase_tracer_file)

    if not phase_structure.paths:
        raise RuntimeError(
            'No valid transition path to the current phase of the Universe')

    analyser = PhaseHistoryAnalyser(potential, phase_structure)
    analyser.analyse(bubble_wall_velocity=bubble_wall_velocity, action_ct=action_ct)
    return analyser.report()


def trace_dof(potential, phase_structure=None, phase_tracer_file=None):
    """
    @param potential Effective potential
    @param phase_structure Parsed phase structure from PT
    @param phase_tracer_file Results file from PT

    @returns DOF as a function of temperature for each phase
    """
    if phase_tracer_file is not None:
        phase_structure = read_phase_tracer(phase_tracer_file=phase_tracer_file)

    data = {}

    for phase in phase_structure.phases:
        t1 = phase.T[0] if phase.T[0] != 0 else phase.T[1]
        T = np.geomspace(t1, phase.T[-1], 1000).tolist()
        phi = [phase.find_phase_at_t(t, potential) for t in T]
        dof = [geff.field_dependent_dof(potential, p, t) for p, t in zip(phi, T)]
        data[str(phase.key)] = {"T": T, "dof": dof}

    return data
