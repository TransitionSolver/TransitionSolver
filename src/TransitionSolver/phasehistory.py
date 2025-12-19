"""
Run PhaseTracer 
===============
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .analysis import geff
from .analysis.phase_history_analysis import PhaseHistoryAnalyser
from . import read_phase_tracer


def _make_report(paths, phase_structure, analysis_metrics):
    """
    @returns Report phase history from TransitionSolver objects
    """
    report = {}
    report['transitions'] = [t.getReport(None)
                             for t in phase_structure.transitions]
    report['paths'] = [p.report() for p in paths]
    report['valid'] = any([p.bValid for p in paths])
    report['analysisTime'] = analysis_metrics.analysisElapsedTime
    return report


def find_phase_history(potential, phase_structure=None, vw=0.9, phase_structure_file=None):
    """
    @param potential Effective potential
    @param phase_structure Phase structure results file from PT
    @param vw Assumed wall velocity

    @returns Report phase history from PhaseTracer output
    """
    if phase_structure_file is not None:
        with open(phase_structure_file) as f:
            phase_structure = read_phase_tracer(f.read())
            
    if not phase_structure.transitionPaths:
        raise RuntimeError(
            'No valid transition path to the current phase of the Universe')

    analyser = PhaseHistoryAnalyser()
    paths, _, analysis_metrics = analyser.analysePhaseHistory_supplied(
        potential, phase_structure)

    return _make_report(paths, phase_structure, analysis_metrics)


def trace_dof(potential, phase_structure=None, phase_structure_file=None):
    """
    @param potential Effective potential
    @param phase_structure_file Phase structure results file from PT

    @returns DOF as a function of temperature for each phase
    """
    if phase_structure_file is not None:
        with open(phase_structure_file) as f:
            phase_structure = read_phase_tracer(f.read())

    data = {}

    for phase in phase_structure.phases:
        t1 = phase.T[0] if phase.T[0] != 0 else phase.T[1]
        T = np.geomspace(t1, phase.T[-1], 1000)
        phi = [phase.findPhaseAtT(t, potential) for t in T]
        dof = [geff.field_dependent_dof(potential, p, t) for p, t in zip(phi, T)]
        data[phase.key] = {"T": T, "dof": dof}

    return data


def load_transition(transition_id, phase_structure_file):
    """
    @param transition_id ID of transition to load from disk
    @param phase_structure_file Phase structure results file from PT

    @returns Transition from data on disk
    """
    with open(phase_structure_file) as f:
        phase_structure = json.load(f)

    for tr in phase_structure['transitions']:
        if tr['id'] == transition_id:
            return tr

    raise RuntimeError(f"Could not find {transition_id} transition")


def plot_action_curve(transition_id, phase_structure_file, ax=None):
    """
    @param transition_id ID of transition to load from disk
    @param phase_structure_file Phase structure results file from PT

    @returns Axes of plot of axis curve
    """
    if ax is None:
        ax = plt.gca()

    transition = load_transition(transition_id, phase_structure_file)

    ax.plot(transition['T'], transition['SonT'], marker='.')
    ax.set_xlabel('$T$')
    ax.set_ylabel('$S(T) / T$')
    return ax
