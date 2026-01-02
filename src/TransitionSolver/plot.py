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

    if transition['Tf'] is not None:
        maxIndex = len(transition['TSubsampleArray'])
        maxIndex = min(len(transition['TSubsampleArray'])-1, maxIndex - (maxIndex - idx_tf)//2)
        physicalVolumeRelative = [100 * (Tf/transition['TSubsampleArray'][i])**3 * transition.Pf[i]
            for i in range(maxIndex+1)]

        ylim = np.array(physicalVolumeRelative[:min(idx_tf+1, maxIndex)]).max(initial=1.)*1.2

        textXOffset = 0.01*(transition['TSubsampleArray'][0] - transition['TSubsampleArray'][maxIndex])
        textY = 0.1

        ax.plot(transition['TSubsampleArray'][:maxIndex+1], physicalVolumeRelative, zorder=3, lw=3.5)

        if transition.TVphysDecr_high is not None and transition.TVphysDecr_low is not None:
            plt.axvspan(transition.TVphysDecr_low, transition.TVphysDecr_high, alpha=0.3, color='r', zorder=-1)
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
        plt.xlim(0, transition['TSubsampleArray'][0])
        plt.ylim(0, ylim)

    return ax

def plot_dvolume(transition_id, phase_structure_file, ax=None):

    if ax is None:
        ax = plt.gca()
        
    transition = load_transition(transition_id, phase_structure_file)

    ylim = np.array(physicalVolume).min(initial=0.)
    ylim *= 1.2 if ylim < 0 else 0.8
    if -0.5 < ylim < 0:
        ylim = -0.5

    textXOffset = 0.01*(transition['TSubsampleArray'][0] - 0)
    textY = ylim + 0.07*(3.5 - ylim)

    ax.plot(transition['TSubsampleArray'], transition.physical_volume, zorder=3, lw=3.5)

    if transition.TVphysDecr_high is not None and transition.TVphysDecr_low is not None:
        plt.axvspan(transition.TVphysDecr_low, transition.TVphysDecr_high, alpha=0.3, color='r', zorder=-1)
    if transition['Tp'] is not None:
        plt.axvline(transition['Tp'], c='g', ls='--', lw=2)
        plt.text(transition['Tp'] + textXOffset, textY, '$T_p$', fontsize=44, horizontalalignment='left')
    if transition['Te'] is not None:
        plt.axvline(transition['Te'], c='b', ls='--', lw=2)
        plt.text(transition['Te'] + textXOffset, textY, '$T_e$', fontsize=44, horizontalalignment='left')
    if transition['Tf'] is not None:
        plt.axvline(transition['Tf'], c='k', ls='--', lw=2)
        plt.text(transition['Tf'] - textXOffset, textY, '$T_f$', fontsize=44, horizontalalignment='right')
    plt.axhline(3., c='gray', ls=':', lw=2, zorder=-1)
    plt.axhline(0., c='gray', ls=':', lw=2, zorder=-1)
    plt.xlabel('$T \,\, \\mathrm{[GeV]}$', fontsize=50)
    plt.ylabel('$\\frac{\\displaystyle d}{\\displaystyle dt} \\mathcal{V}_{\\mathrm{phys}} \,\, \\mathrm{[GeV]}$',
        fontsize=50, labelpad=20)
    plt.xlim(0, transition['TSubsampleArray'][0])
    plt.ylim(ylim, 3.5)

    return ax

def plot_vw(transition_id, phase_structure_file, ax=None):

    if ax is None:
        ax = plt.gca()

    transition = load_transition(transition_id, phase_structure_file)

    ax.plot(transition['TSubampleArray'], transition['vw_samples'])
    plt.xlabel('T')
    plt.ylabel('$v_w$')
    plt.ylim(0, 1)
    return ax


def plot_gamma(transition_id, phase_structure_file, ax=None):

    if ax is None:
        ax = plt.gca()
        
    transition = load_transition(transition_id, phase_structure_file)

    ax.plot(transition['TSubsampleArray'], transition.gamma)
    ax.plot(transition['TSubsampleArray'], transition.gamma_eff)

    plt.xlabel('$T$', fontsize=24)
    plt.ylabel('$\\Gamma(T)$', fontsize=24)
    if transition.TGammaMax is not None: plt.axvline(transition.TGammaMax, c='g', ls=':')
    if transition.Tmin is not None: plt.axvline(transition.Tmin, c='r', ls=':')
    if transition['Tp'] is not None: plt.axvline(transition['Tp'], c='g', ls=':')
    if transition['Te'] is not None: plt.axvline(transition['Te'], c='b', ls=':')
    if transition['Tf'] is not None: plt.axvline(transition['Tf'], c='k', ls=':')
    plt.legend(['$\\mathrm{standard}$', '$\\mathrm{effective}$'], fontsize=24)

    return ax

def plot_bubble_number(transition_id, phase_structure_file, ax=None):

    if ax is None:
        ax = plt.gca()

    transition = load_transition(transition_id, phase_structure_file)

    ax.plot(transition['TSubsampleArray'], bubbleNumberDensity, linewidth=2.5)
    plt.xlabel('$T \, \\mathrm{[GeV]}$', fontsize=24)
    plt.ylabel('$n_B(T)$', fontsize=24)
    if transition['Tn'] is not None: plt.axvline(transition['Tn'], c='r', ls=':')
    if transition['Tp'] is not None: plt.axvline(transition['Tp'], c='g', ls=':')
    if transition['Te'] is not None: plt.axvline(transition['Te'], c='b', ls=':')
    if transition['Tf'] is not None: plt.axvline(transition['Tf'], c='k', ls=':')
    return ax
    
def plot_bubble_radius(transition_id, phase_structure_file, ax=None):

    if ax is None:
        ax = plt.gca()

    transition = load_transition(transition_id, phase_structure_file)

    highTempIndex = 1

    # Search for the highest temperature for which the mean bubble separation is not larger than it is at the
    # lowest sampled temperature.
    for i in range(1, len(transition['TSubsampleArray'])):
        if transition.meanBubbleSeparationArray[i] <= transition.meanBubbleSeparationArray[-1]:
            highTempIndex = i
            break

    # If the mean bubble separation is always larger than it is at the lowest sampled temperature, plot the
    # entire range of sampled temperatures.
    if highTempIndex == len(transition['TSubsampleArray'])-1:
        highTempIndex = 0

    ax.plot(transition['TSubsampleArray'], transition.meanBubbleRadiusArray, linewidth=2.5)
    ax.plot(transition['TSubsampleArray'], transitionmeanBubbleSeparationArray, linewidth=2.5)
    plt.xlabel('$T \, \\mathrm{[GeV]}$', fontsize=24)

    plt.legend(['$\\overline{R}_B(T)$', '$R_*(T)$'], fontsize=24)
    if Tn > 0: plt.axvline(Tn, c='r', ls=':')
    if Tp > 0: plt.axvline(Tp, c='g', ls=':')
    if Te > 0: plt.axvline(Te, c='b', ls=':')
    if Tf > 0: plt.axvline(Tf, c='k', ls=':')
    plt.xlim(transition['TSubsampleArray'][-1], transition['TSubsampleArray'][highTempIndex])
    plt.ylim(0, 1.2*max(transitionmeanBubbleSeparationArray[-1], transitionmeanBubbleRadiusArray[-1]))

    return ax

def plot_action_curve(transition_id, phase_structure_file, ax=None):

    if ax is None:
        ax = plt.gca()

    transition = load_transition(transition_id, phase_structure_file)

    highTempIndex = 0
    lowTempIndex = len(transition.SonT)

    minAction = min(transition.SonT)
    maxAction = actionSampler.maxSonTThreshold

    # Search for the lowest temperature for which the action is not significantly larger than the maximum
    # significant action.
    for i in range(len(transition.SonT)):
        if transition.SonT[i] <= maxAction:
            highTempIndex = i
            break

    # Search for the lowest temperature for which the action is not significantly larger than the maximum
    # significant action.
    for i in range(len(transition.SonT)-1, -1, -1):
        if transition.SonT[i] <= maxAction:
            lowTempIndex = i
            break

    plt.figure(figsize=(12,8))
    ax.plot(transition['TSubsampleArray'], actionSampler.subSonT, linewidth=2.5)

    plt.scatter(actionSampler.T, transition.SonT)
    if Tn > -1:
        plt.axvline(transition['Tn'], c='r', ls=':')
        plt.axhline(transition.SonTn, c='r', ls=':')
    if Tp > -1:
        plt.axvline(transition['Tp'], c='g', ls=':')
        plt.axhline(transition.SonTp, c='g', ls=':')
    if Te > -1:
        plt.axvline(transition['Te'], c='b', ls=':')
        plt.axhline(transition.SonTe, c='b', ls=':')
    if Tf > -1:
        plt.axvline(transition['Tf'], c='k', ls=':')
        plt.axhline(transition.SonTf, c='k', ls=':')
    plt.minorticks_on()
    plt.grid(visible=True, which='major', color='k', linestyle='--')
    plt.grid(visible=True, which='minor', color='gray', linestyle=':')
    plt.xlabel('$T \, \\mathrm{[GeV]}$', fontsize=24)
    plt.ylabel('$S(T)$', fontsize=24)

    plt.xlim(transition.T[lowTempIndex], transition.T[highTempIndex])
    plt.ylim(minAction - 0.05*(maxAction - minAction), maxAction)

    return ax

def plot_bubble_number(transition_id, phase_structure_file, ax=None):

    if ax is None:
        ax = plt.gca()

    transition = load_transition(transition_id, phase_structure_file)

    # Number of bubbles plotted over entire sampled temperature range, using log scale for number of bubbles.

    ax.plot(transition['TSubsampleArray'], transition.totalNumBubblesCorrected, linewidth=2.5)
    ax.plot(transition['TSubsampleArray'], transition.totalNumBubbles, linewidth=2.5)

    if transition['Tn']: plt.axvline(transition['Tn'], c='r', ls=':')
    if transition['Tp']: plt.axvline(transition['Tp'], c='g', ls=':')
    if transition['Te']: plt.axvline(transition['Te'], c='b', ls=':')
    if transition['Tf']: plt.axvline(transition['Tf'], c='k', ls=':')

    plt.yscale('log')
    plt.legend(['$N(T)$', '$N^{\\mathrm{ext}}(T)$'], fontsize=24)
    plt.xlabel('$T \, \\mathrm{[GeV]}$', fontsize=24)

    return ax

def plot_p_f(transition_id, phase_structure_file, ax=None):

    if ax is None:
        ax = plt.gca()

    ax.plot(transition['TSubsampleArray'], transition.Pf, linewidth=2.5)
    if transition['Tn']: plt.axvline(transition['Tn'], c='r', ls=':')
    if transition['Tp']: plt.axvline(transition['Tp'], c='g', ls=':')
    if transition['Te']: plt.axvline(transition['Te'], c='b', ls=':')
    if transition['Tf']: plt.axvline(transition['Tf'], c='k', ls=':')
    plt.xlabel('$T \, \\mathrm{[GeV]}$', fontsize=40)
    plt.ylabel('$P_f(T)$', fontsize=40)

    return ax
