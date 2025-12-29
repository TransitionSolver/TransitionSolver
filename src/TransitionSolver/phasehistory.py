"""
Run PhaseTracer 
===============
"""

import json

import numpy as np
import matplotlib.pyplot as plt

from . import geff
from .analysis.phase_history_analysis import PhaseHistoryAnalyser
from . import read_phase_tracer


def _make_report(paths, phase_structure, analysis_metrics):
    """
    @returns Report phase history from TransitionSolver objects
    """
    report = {}
    report['transitions'] = [t.report(None)
                             for t in phase_structure.transitions]
    report['paths'] = [p.report() for p in paths]
    report['valid'] = any(p.is_valid for p in paths)
    report['analysisTime'] = analysis_metrics.analysisElapsedTime
    return report


def find_phase_history(potential, phase_structure=None, vw=0.9, phase_tracer_file=None):
    """
    @param potential Effective potential
    @param phase_structure Parsed phase structure from PT
    @param vw Assumed wall velocity
    @param phase_tracer_file Results file from PT

    @returns Report phase history from PhaseTracer output
    """
    if phase_tracer_file is not None:
        phase_structure = read_phase_tracer(phase_tracer_file=phase_tracer_file)

    if not phase_structure.paths:
        raise RuntimeError(
            'No valid transition path to the current phase of the Universe')

    analyser = PhaseHistoryAnalyser()
    paths, _, analysis_metrics = analyser.analysePhaseHistory_supplied(
        potential, phase_structure)

    return _make_report(paths, phase_structure, analysis_metrics)


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
        T = np.geomspace(t1, phase.T[-1], 1000)
        phi = [phase.find_phase_at_t(t, potential) for t in T]
        dof = [geff.field_dependent_dof(potential, p, t) for p, t in zip(phi, T)]
        data[phase.key] = {"T": T, "dof": dof}

    return data


def load_transition(transition_id, phase_structure_file):
    """
    @param transition_id ID of transition to load from disk
    @param phase_structure_file Phase structure in JSON format

    @returns Transition from data on disk
    """
    with open(phase_structure_file, encoding="utf8") as f:
        phase_structure = json.load(f)

    for tr in phase_structure['transitions']:
        if tr['id'] == transition_id:
            return tr

    raise RuntimeError(f"Could not find {transition_id} transition")


def plot_action_curve(transition_id, phase_structure_file, ax=None):
    """
    @param transition_id ID of transition to load from disk
    @param phase_structure_file Phase structure in JSON format

    @returns Axes of plot of axis curve
    """
    if ax is None:
        ax = plt.gca()

    transition = load_transition(transition_id, phase_structure_file)

    ax.plot(transition['T'], transition['SonT'], marker='.')
    ax.set_xlabel('$T$')
    ax.set_ylabel('$S(T) / T$')
    return ax
