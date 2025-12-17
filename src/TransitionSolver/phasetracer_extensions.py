"""
PhaseTracer extensions
======================
"""

import json

import numpy as np
import matplotlib.pyplot as plt


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
