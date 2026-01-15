"""
Make plots from results on disk
===============================
"""

import json_numpy as json
import matplotlib.pyplot as plt
import numpy as np


def load_transition(transition_id, phase_structure):
    """
    @param transition_id ID of transition
    @param phase_structure Phase structure

    @returns Transition data
    """
    for tr in phase_structure['transitions']:
        if tr['id'] == transition_id:
            return tr

    raise RuntimeError(f"Could not find {transition_id} transition")


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
    if 'TGammaMax' in transition:
        add_labeled_vline(ax, transition['TGammaMax'], r"$T_\Gamma$", 'k')
    if 'Tn' in transition:
        add_labeled_vline(ax, transition['Tn'], "$T_n$", 'r')
    if 'Tp' in transition:
        add_labeled_vline(ax, transition['Tp'], "$T_p$", 'g')
    if 'Te' in transition:
        add_labeled_vline(ax, transition['Te'], "$T_e$", 'b')
    if 'Tf' in transition:
        add_labeled_vline(ax, transition['Tf'], "$T_f$", 'm')


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

    transition = load_transition(transition_id, phase_structure)

    ax.plot(transition['TSubSampleArray'],
            transition['physical_volume'], zorder=3)

    if 'TVphysDecr_high' in transition and 'TVphysDecr_low' in transition:
        ax.axvspan(transition['TVphysDecr_low'],
                   transition['TVphysDecr_high'], alpha=0.3, color='r', zorder=-1)
                   
    add_labeled_vlines(transition, ax)
    add_labeled_hline(ax, 3., "3", 'gray')
    add_labeled_hline(ax, 0., "0", 'gray')

    ax.set_xlabel('$T$ (GeV)')
    ax.set_ylabel(
        '$\\frac{d}{dt} \\mathcal{V}_{\\mathrm{phys}}$ (GeV)')

    if show:
        plt.show()

    return ax


def plot_vw(transition_id, phase_structure=None, phase_structure_file=None, ax=None, show=False):
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

    transition = load_transition(transition_id, phase_structure)

    ax.plot(transition['TSubSampleArray'], transition['vw_samples'])
 
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

    transition = load_transition(transition_id, phase_structure)

    ax.plot(transition['TSubSampleArray'],
            transition['gamma'], label="Standard")
    ax.plot(transition['TSubSampleArray'],
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

    transition = load_transition(transition_id, phase_structure)

    ax.plot(transition['TSubSampleArray'],
            transition['meanBubbleRadiusArray'])
            
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

    transition = load_transition(transition_id, phase_structure)

    ax.plot(transition['TSubSampleArray'],
            transition['meanBubbleSeparationArray'])
            
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

    transition = load_transition(transition_id, phase_structure)

    # Number of bubbles plotted over entire sampled temperature range, using log scale for number of bubbles.

    ax.plot(transition['TSubSampleArray'],
            transition['totalNumBubblesCorrected'], label='$N(T)$')
    ax.plot(transition['TSubSampleArray'],
            transition['totalNumBubbles'], ls="--", label='$N^{\\mathrm{ext}}(T)$')

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

    transition = load_transition(transition_id, phase_structure)

    ax.plot(transition['TSubSampleArray'], transition['Pf'])

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

    transition = load_transition(transition_id, phase_structure)

    ax.plot(transition['T'], transition['SonT'])

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

    transitions = [t for t in phase_structure['transitions'] if t['analysed']]

    if not transitions:
        return plt.figure()

    plotters = [plot_volume, plot_vw, plot_gamma, plot_bubble_radius, plot_bubble_separation, plot_bubble_number, plot_action_curve, plot_pf]
    fig, ax = plt.subplots(len(plotters), len(transitions), constrained_layout=False, sharex='col', figsize=(10 * len(transitions), 6 * len(plotters)))
    ax = np.reshape(ax, (len(plotters), len(transitions)))

    for x, tr in enumerate(transitions):
        for y, p in enumerate(plotters):
            this_ax = ax[y, x]
            p(tr['id'], phase_structure, ax=this_ax)
            this_ax.set_xlabel(None)

    fig.supxlabel("$T$ (GeV)", y=0.005)
    fig.set_tight_layout(True)

    if show:
        plt.show()

    return fig
