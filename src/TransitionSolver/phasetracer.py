"""
Run PhaseTracer 
===============
"""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .analysis import geff
from .analysis.phase_structure import load_data
from .analysis.phase_history_analysis import PhaseHistoryAnalyser


PT_HOME = Path(os.getenv("PHASETRACER", Path.home() / ".PhaseTracer"))
PT_LIB = PT_HOME / "lib" / "libphasetracer.so"
PT_BIN = PT_HOME / "bin"
PT_UNIT_TEST = PT_BIN / "unit_tests"
WINDOWS = os.name == 'nt'
PT_DATA = Path(os.path.dirname(os.path.abspath(__file__))) / "data"


def pt_run(program, point, output_dir="output", **kwargs):
    """
    Run a PT program

    @param name Name of executable
    @param output_dir Directory for output
    @param point Parameter point

    @returns Phase structure file from PT
    """
    program = PT_BIN / program
    assert program.is_file()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    point_file = output_dir / "parameter_point.txt"
    np.savetxt(point_file, np.matrix(point))

    command = [program, point_file, output_dir]

    if WINDOWS:
        command.insert(0, "wsl")

    subprocess.check_call(command, **kwargs)

    phase_structure_file = output_dir / 'phase_structure.dat'
    assert phase_structure_file.is_file()
    return phase_structure_file


def make_report(paths, phase_structure, analysis_metrics):
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


def phase_history(potential, phase_structure=None, vw=0.9, phase_structure_file=None):
    """
    @param potential Effective potential
    @param phase_structure Phase structure results file from PT
    @param vw Assumed wall velocity

    @returns Report phase history from PhaseTracer output
    """
    if phase_structure_file is not None:
        phase_structure = load_data(phase_structure_file)[1]
            
    if not phase_structure.transitionPaths:
        raise RuntimeError(
            'No valid transition path to the current phase of the Universe')

    analyser = PhaseHistoryAnalyser()
    paths, _, analysis_metrics = analyser.analysePhaseHistory_supplied(
        potential, phase_structure)

    return make_report(paths, phase_structure, analysis_metrics)


def trace_dof(potential, phase_structure_file):
    """
    @param potential Effective potential
    @param phase_structure_file Phase structure results file from PT

    @returns DOF as a function of temperature for each phase
    """
    phase_structure = load_data(phase_structure_file)[1]

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
