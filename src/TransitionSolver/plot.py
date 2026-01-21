"""
Make plots from results on disk
===============================
"""

import json
import matplotlib.pyplot as plt
import numpy as np


def add_labeled_vline(ax, x, label, color):
    transform = ax.get_xaxis_transform()
    y = 0.8
    ax.axvline(x, c=color, ls=':')
    ax.text(x, y, label,  c=color, transform=transform, ha='center', bbox={'fc': 'w', 'ec': color})


def add_labeled_hline(ax, y, label, color):
    transform = ax.get_yaxis_transform()
    x = 0.1
    ax.axhline(y, c=color, ls=':')
    ax.text(x, y, label,  c=color, transform=transform, va='center', bbox={'fc': 'w', 'ec': color})


def add_labeled_vlines(transition, ax):
    if 'T_gamma' in transition:
        add_labeled_vline(ax, transition['T_gamma'], r"$T_\Gamma$", 'k')
    if 'T_n' in transition:
        add_labeled_vline(ax, transition['T_n'], "$T_n$", 'r')
    if 'T_p' in transition:
        add_labeled_vline(ax, transition['T_p'], "$T_p$", 'g')
    if 'T_e' in transition:
        add_labeled_vline(ax, transition['T_e'], "$T_e$", 'b')
    if 'T_f' in transition:
        add_labeled_vline(ax, transition['T_f'], "$T_f$", 'm')


def plot_volume(transition_id, phase_structure=None, phase_structure_file=None, ax=None, show=False):
    """
    @param transition_id ID of transition
    @param phase_structure Phase structure
    @param phase_structure_file Phase structure in JSON format

    @returns Axes of plot of change in volume
    """
    if phase_structure_file is not None:
        with open(phase_structure_file, encoding="utf8") as f:
            phase_structure = json.load(f)

    if ax is None:
        ax = plt.gca()

    transition = phase_structure['transitions'][transition_id]

    ax.plot(transition['T'],
            transition['deriv_physical_volume'], zorder=3)

    if 'T_decreasing_v_phys' in transition and 'T_increasing_v_phys' in transition:
        ax.axvspan(transition['T_decreasing_v_phys'],
                   transition['T_increasing_v_phys'], alpha=0.3, color='r', zorder=-1)

    add_labeled_vlines(transition, ax)
    add_labeled_hline(ax, 3., "3", 'gray')
    add_labeled_hline(ax, 0., "0", 'gray')

    ax.set_xlabel('$T$ (GeV)')
    ax.set_ylabel(
        '$\\frac{d}{dt} \\mathcal{V}_{\\mathrm{phys}}$ (GeV)')

    if show:
        plt.show()

    return ax


def plot_bubble_wall_velocity(transition_id, phase_structure=None, phase_structure_file=None, ax=None, show=False):
    """
    @param transition_id ID of transition
    @param phase_structure Phase structure
    @param phase_structure_file Phase structure in JSON format

    @returns Axes of plot of bubble wall velocity
    """
    if phase_structure_file is not None:
        with open(phase_structure_file, encoding="utf8") as f:
            phase_structure = json.load(f)

    if ax is None:
        ax = plt.gca()

    transition = phase_structure['transitions'][transition_id]

    ax.plot(transition['T'], transition['bubble_wall_velocity'])

    add_labeled_vlines(transition, ax)

    ax.set_xlabel('$T$ (GeV)')
    ax.set_ylabel('Bubble wall velocity, $v_w$')

    if show:
        plt.show()

    return ax


def plot_gamma(transition_id, phase_structure=None, phase_structure_file=None, ax=None, show=False):
    """
    @param transition_id ID of transition
    @param phase_structure Phase structure
    @param phase_structure_file Phase structure in JSON format

    @returns Axes of plot of transition rate Gamma
    """
    if phase_structure_file is not None:
        with open(phase_structure_file, encoding="utf8") as f:
            phase_structure = json.load(f)

    if ax is None:
        ax = plt.gca()

    transition = phase_structure['transitions'][transition_id]

    ax.plot(transition['T'],
            transition['gamma'], label="Standard")
    ax.plot(transition['T'],
            transition['gamma_eff'], label="Effective")

    ax.set_xlabel('$T$ (GeV)')
    ax.set_ylabel('Transition rate, $\\Gamma(T)$')

    add_labeled_vlines(transition, ax)

    ax.legend()

    if show:
        plt.show()

    return ax


def plot_bubble_radius(transition_id, phase_structure=None, phase_structure_file=None, ax=None, show=False):
    """
    @param transition_id ID of transition
    @param phase_structure Phase structure
    @param phase_structure_file Phase structure in JSON format

    @returns Axes of plot of bubble radius
    """
    if phase_structure_file is not None:
        with open(phase_structure_file, encoding="utf8") as f:
            phase_structure = json.load(f)

    if ax is None:
        ax = plt.gca()

    transition = phase_structure['transitions'][transition_id]

    ax.plot(transition['T'],
            transition['bubble_radius'])

    ax.set_yscale('log')
    ax.set_xlabel('$T$ (GeV)')
    ax.set_ylabel('Bubble radius, $\\overline{R}_B(T)$')

    add_labeled_vlines(transition, ax)

    if show:
        plt.show()

    return ax


def plot_bubble_separation(transition_id, phase_structure=None, phase_structure_file=None, ax=None, show=False):
    """
    @param transition_id ID of transition
    @param phase_structure Phase structure
    @param phase_structure_file Phase structure in JSON format

    @returns Axes of plot of bubble separation
    """
    if phase_structure_file is not None:
        with open(phase_structure_file, encoding="utf8") as f:
            phase_structure = json.load(f)

    if ax is None:
        ax = plt.gca()

    transition = phase_structure['transitions'][transition_id]

    ax.plot(transition['T'],
            transition['bubble_separation'])

    ax.set_yscale('log')
    ax.set_xlabel('$T$ (GeV)')
    ax.set_ylabel('Bubble separation, $R_*(T)$')

    add_labeled_vlines(transition, ax)

    if show:
        plt.show()

    return ax


def plot_bubble_number(transition_id, phase_structure=None, phase_structure_file=None, ax=None, show=False):
    """
    @param transition_id ID of transition
    @param phase_structure Phase structure
    @param phase_structure_file Phase structure in JSON format

    @returns Axes of plot of number of bubbles
    """
    if phase_structure_file is not None:
        with open(phase_structure_file, encoding="utf8") as f:
            phase_structure = json.load(f)

    if ax is None:
        ax = plt.gca()

    transition = phase_structure['transitions'][transition_id]

    # Number of bubbles plotted over entire sampled temperature range, using log scale for number of bubbles.

    ax.plot(transition['T'],
            transition['bubble_num_bar'], label='$N(T)$')
    ax.plot(transition['T'],
            transition['bubble_num'], ls="--", label='$N^{\\mathrm{ext}}(T)$')

    add_labeled_vlines(transition, ax)

    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel('$T$ (GeV)')
    ax.set_ylabel('Number of bubbles')

    if show:
        plt.show()

    return ax


def plot_pf(transition_id, phase_structure=None, phase_structure_file=None, ax=None, show=False):
    """
    @param transition_id ID of transition
    @param phase_structure Phase structure
    @param phase_structure_file Phase structure in JSON format

    @returns Axes of plot of false vacuum fraction
    """
    if phase_structure_file is not None:
        with open(phase_structure_file, encoding="utf8") as f:
            phase_structure = json.load(f)

    if ax is None:
        ax = plt.gca()

    transition = phase_structure['transitions'][transition_id]

    ax.plot(transition['T'], transition['Pf'])

    add_labeled_vlines(transition, ax)
    ax.set_yscale('log')
    ax.set_xlabel('$T$ (GeV)')
    ax.set_ylabel('False vacuum fraction, $P_f(T)$')

    if show:
        plt.show()

    return ax


def plot_action_curve(transition_id, phase_structure=None, phase_structure_file=None, ax=None, show=False):
    """
    @param transition_id ID of transition
    @param phase_structure Phase structure
    @param phase_structure_file Phase structure in JSON format

    @returns Axes of plot of action over temperature
    """
    if phase_structure_file is not None:
        with open(phase_structure_file, encoding="utf8") as f:
            phase_structure = json.load(f)

    if ax is None:
        ax = plt.gca()

    transition = phase_structure['transitions'][transition_id]

    ax.plot(transition['T'], transition['action_3d'])

    add_labeled_vlines(transition, ax)

    ax.set_xlabel('$T$ (GeV)')
    ax.set_ylabel('$S(T) / T$')

    if show:
        plt.show()

    return ax


def plot_summary(phase_structure=None, phase_structure_file=None, show=False):
    """
    @param transition_id ID of transition
    @param phase_structure Phase structure
    @param phase_structure_file Phase structure in JSON format

    @returns Axes of plots
    """
    if phase_structure_file is not None:
        with open(phase_structure_file, encoding="utf8") as f:
            phase_structure = json.load(f)

    transitions = {k: v for k, v  in phase_structure['transitions'].items() if v['analysed']}

    if not transitions:
        return plt.figure()

    plotters = [plot_volume, plot_bubble_wall_velocity, plot_gamma, plot_bubble_radius, plot_bubble_separation, plot_bubble_number, plot_action_curve, plot_pf]
    fig, ax = plt.subplots(len(plotters), len(transitions), constrained_layout=False, sharex='col', figsize=(10 * len(transitions), 6 * len(plotters)))
    ax = np.reshape(ax, (len(plotters), len(transitions)))

    for x, key in enumerate(transitions):
        for y, p in enumerate(plotters):
            this_ax = ax[y, x]
            p(key, phase_structure, ax=this_ax)
            this_ax.set_xlabel(None)

    fig.supxlabel("$T$ (GeV)", y=0.005)
    fig.set_tight_layout(True)

    if show:
        plt.show()

    return fig
