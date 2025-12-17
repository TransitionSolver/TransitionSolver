"""
PhaseTracer extensions
======================
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .analysis import geff


def trace_dof(potential, phase_structure):
    """
    @param potential Effective potential
    @param phase_structure Phase structure from PT

    @returns DOF as a function of temperature for each phase
    """
    data = {}

    for phase in phase_structure.phases:
        t1 = phase.T[0] if phase.T[0] != 0 else phase.T[1]
        T = np.geomspace(t1, phase.T[-1], 1000)
        phi = [phase.findPhaseAtT(t, potential) for t in T]
        dof = [geff.field_dependent_dof(potential, p, t) for p, t in zip(phi, T)]
        data[phase.key] = {"T": T, "dof": dof}

    return data


def _load_transition(transition_id, phase_history_file):
    """
    @param transition_id ID of transition to load from disk
    @param phase_history_file Phase history file

    @returns Transition from data on disk
    """
    with open(phase_history_file) as f:
        phase_history = json.load(f)

    for tr in phase_history['transitions']:
        if tr['id'] == transition_id:
            return tr

    raise RuntimeError(f"Could not find {transition_id} transition")


def plot_action_curve(transition_id, phase_history_file, ax=None):
    """
    @param transition_id ID of transition to load from disk
    @param phase_history_file Phase history file

    @returns Axes of plot of axis curve
    """
    if ax is None:
        ax = plt.gca()

    transition = _load_transition(transition_id, phase_history_file)

    ax.plot(transition['T'], transition['SonT'], marker='.')
    ax.set_xlabel('$T$')
    ax.set_ylabel('$S(T) / T$')
    return ax
