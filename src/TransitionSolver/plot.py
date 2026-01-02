"""
Make plots from results on disk
===============================
"""

import matplotlib.pyplot as plt
import json


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


def plot_volume(transition_id, phase_structure_file, ax=None):

    if ax is None:
        ax = plt.gca()

    transition = load_transition(transition_id, phase_structure_file)

    if 'Tf' not in transition['Tf']:
        return ax
        
    maxIndex = len(transition['TSubsampleArray'])
    maxIndex = min(len(transition['TSubsampleArray'])-1, maxIndex - (maxIndex - idx_tf)//2)
    physicalVolumeRelative = [100 * (Tf/transition['TSubsampleArray'][i])**3 * transition['Pf'][i]
        for i in range(maxIndex+1)]


    textXOffset = 0.01*(transition['TSubsampleArray'][0] - transition['TSubsampleArray'][maxIndex])
    textY = 0.1

    ax.plot(transition['TSubsampleArray'][:maxIndex+1], physicalVolumeRelative, zorder=3, lw=3.5)

    if transition['TVphysDecr_high'] is not None and transition['TVphysDecr_low'] is not None:
        plt.axvspan(transition['TVphysDecr_low'], transition['TVphysDecr_high'], alpha=0.3, color='r', zorder=-1)
    if transition['Tp'] is not None:
        plt.axvline(transition['Tp'], c='g', ls='--', lw=2)
        plt.text(Tp + textXOffset, textY, '$T_p$', fontsize=44, horizontalalignment='left')
    if transition['Te'] is not None:
        plt.axvline(transition['Te'], c='b', ls='--', lw=2)
        plt.text(transition['Te'] + textXOffset, textY, '$T_e$', fontsize=44, horizontalalignment='left')
    if transition['Tf'] is not None:
        plt.axvline(transition['Tf'], c='k', ls='--', lw=2)
        plt.text(transition['Tf'] - textXOffset, textY, '$T_f$', fontsize=44, horizontalalignment='right')
    plt.axhline(1., c='gray', ls=':', lw=2, zorder=-1)
    plt.xlabel('$T \,\, \\mathrm{[GeV]}$', fontsize=52)
    plt.ylabel('$\\mathcal{V}_{\\mathrm{phys}}(T)/\\mathcal{V}_{\\mathrm{phys}}(T_f)$', fontsize=52,
        labelpad=20)


    return ax

def plot_dvolume(transition_id, phase_structure_file, ax=None):

    if ax is None:
        ax = plt.gca()
        
    transition = load_transition(transition_id, phase_structure_file)

    ax.plot(transition['TSubsampleArray'], transition['physical_volume'], zorder=3)

    if 'TVphysDecr_high' in transition and 'TVphysDecr_low' in transition:
        plt.axvspan(transition['TVphysDecr_low'], transition['TVphysDecr_high'], alpha=0.3, color='r', zorder=-1)
    if 'Tn' in transition:
        plt.axvline(transition['Tn'], c='r', ls=':')
    if 'Tp' in transition:
        plt.axvline(transition['Tp'], c='g', ls=':')
    if 'Te' in transition:
        plt.axvline(transition['Te'], c='b', ls=':')
    if 'Tf' in transition:
        plt.axvline(transition['Tf'], c='k', ls=':')

    plt.axhline(3., c='gray', ls=':', zorder=-1)
    plt.axhline(0., c='gray', ls=':', zorder=-1)
    
    plt.xlabel('$T \,\, \\mathrm{[GeV]}$')
    plt.ylabel('$\\frac{\\displaystyle d}{\\displaystyle dt} \\mathcal{V}_{\\mathrm{phys}} \,\, \\mathrm{[GeV]}$')

    return ax

def plot_vw(transition_id, phase_structure_file, ax=None):

    if ax is None:
        ax = plt.gca()

    transition = load_transition(transition_id, phase_structure_file)

    ax.plot(transition['TSubampleArray'], transition['vw_samples'])
    plt.xlabel('$T$')
    plt.ylabel('$v_w$')
    return ax


def plot_gamma(transition_id, phase_structure_file, ax=None):

    if ax is None:
        ax = plt.gca()
        
    transition = load_transition(transition_id, phase_structure_file)

    ax.plot(transition['TSubsampleArray'], transition['gamma'], label="Standard")
    ax.plot(transition['TSubsampleArray'], transition['gamma_eff'], label="Effective")

    plt.xlabel('$T$')
    plt.ylabel('$\\Gamma(T)$')

    if 'Tn' in transition:
        plt.axvline(transition['TGammaMax'], c='r', ls=':') 
    if 'Tn' in transition:
        plt.axvline(transition['Tmin'], c='r', ls=':')
    if 'Tn' in transition:
        plt.axvline(transition['Tn'], c='r', ls=':')
    if 'Tp' in transition:
        plt.axvline(transition['Tp'], c='g', ls=':')
    if 'Te' in transition:
        plt.axvline(transition['Te'], c='b', ls=':')
    if 'Tf' in transition:
        plt.axvline(transition['Tf'], c='k', ls=':')

    ax.legend()

    return ax

def plot_bubble_number(transition_id, phase_structure_file, ax=None):

    if ax is None:
        ax = plt.gca()

    transition = load_transition(transition_id, phase_structure_file)

    ax.plot(transition['TSubsampleArray'], bubbleNumberDensity)

    plt.xlabel('$T \, \\mathrm{[GeV]}$')
    plt.ylabel('$n_B(T)$')
    
    if 'Tn' in transition:
        plt.axvline(transition['Tn'], c='r', ls=':')
    if 'Tp' in transition:
        plt.axvline(transition['Tp'], c='g', ls=':')
    if 'Te' in transition:
        plt.axvline(transition['Te'], c='b', ls=':')
    if 'Tf' in transition:
        plt.axvline(transition['Tf'], c='k', ls=':')
        
    return ax
    
def plot_bubble_radius(transition_id, phase_structure_file, ax=None):

    if ax is None:
        ax = plt.gca()

    transition = load_transition(transition_id, phase_structure_file)

    highTempIndex = 1

    # Search for the highest temperature for which the mean bubble separation is not larger than it is at the
    # lowest sampled temperature.
    for i in range(1, len(transition['TSubsampleArray'])):
        if transition['meanBubbleSeparationArray'][i] <= transition['meanBubbleSeparationArray'][-1]:
            highTempIndex = i
            break

    # If the mean bubble separation is always larger than it is at the lowest sampled temperature, plot the
    # entire range of sampled temperatures.
    if highTempIndex == len(transition['TSubsampleArray'])-1:
        highTempIndex = 0

    ax.plot(transition['TSubsampleArray'], transition['meanBubbleRadiusArray'])
    ax.plot(transition['TSubsampleArray'], transition['meanBubbleSeparationArray'])
    plt.xlabel('$T \, \\mathrm{[GeV]}$')

    plt.legend(['$\\overline{R}_B(T)$', '$R_*(T)$'])

    if 'Tn' in transition:
        plt.axvline(transition['Tn'], c='r', ls=':')
    if 'Tp' in transition:
        plt.axvline(transition['Tp'], c='g', ls=':')
    if 'Te' in transition:
        plt.axvline(transition['Te'], c='b', ls=':')
    if 'Tf' in transition:
        plt.axvline(transition['Tf'], c='k', ls=':')


    return ax

def plot_action_curve(transition_id, phase_structure_file, ax=None):

    if ax is None:
        ax = plt.gca()

    transition = load_transition(transition_id, phase_structure_file)

    plt.plot(transition['T'], transition['SonT'])
    
    if 'Tn' in transition:
        plt.axvline(transition['Tn'], c='r', ls=':')
        plt.axhline(transition['SonTn'], c='r', ls=':')
    if 'Tp' in transition:
        plt.axvline(transition['Tp'], c='g', ls=':')
        plt.axhline(transition['SonTp'], c='g', ls=':')
    if 'Te' in transition:
        plt.axvline(transition['Te'], c='b', ls=':')
        plt.axhline(transition['SonTe'], c='b', ls=':')
    if 'Tf' in transition:
        plt.axvline(transition['Tf'], c='k', ls=':')
        plt.axhline(transition['SonTf'], c='k', ls=':')

    plt.xlabel('$T \, \\mathrm{[GeV]}$')
    plt.ylabel('$S(T)$')

    return ax

def plot_bubble_number(transition_id, phase_structure_file, ax=None):

    if ax is None:
        ax = plt.gca()

    transition = load_transition(transition_id, phase_structure_file)

    # Number of bubbles plotted over entire sampled temperature range, using log scale for number of bubbles.

    ax.plot(transition['TSubsampleArray'], transition['totalNumBubblesCorrected'], label='$N(T)$')
    ax.plot(transition['TSubsampleArray'], transition['totalNumBubbles'], label='$N^{\\mathrm{ext}}(T)$')

    if 'Tn' in transition:
        plt.axvline(transition['Tn'], c='r', ls=':')
    if 'Tp' in transition:
        plt.axvline(transition['Tp'], c='g', ls=':')
    if 'Te' in transition:
        plt.axvline(transition['Te'], c='b', ls=':')
    if 'Tf' in transition:
        plt.axvline(transition['Tf'], c='k', ls=':')

    plt.yscale('log')
    ax.legend()
    plt.xlabel('$T \, \\mathrm{[GeV]}$')

    return ax

def plot_p_f(transition_id, phase_structure_file, ax=None):

    if ax is None:
        ax = plt.gca()

    ax.plot(transition['TSubsampleArray'], transition['Pf']5)
    
    if 'Tn' in transition:
        plt.axvline(transition['Tn'], c='r', ls=':')
    if 'Tp' in transition:
        plt.axvline(transition['Tp'], c='g', ls=':')
    if 'Te' in transition:
        plt.axvline(transition['Te'], c='b', ls=':')
    if 'Tf' in transition:
        plt.axvline(transition['Tf'], c='k', ls=':')
        
    plt.xlabel('$T \, \\mathrm{[GeV]}$', fontsize=40)
    plt.ylabel('$P_f(T)$', fontsize=40)

    return ax
