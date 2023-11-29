"""
Plot GW benchmarks
==================
"""

import numpy as np
import matplotlib.pyplot as plt

from gws.gw_analyser import GWAnalyser
from gws.detectors.lisa import LISA
from models.Archil_model import SMplusCubic
from examples.scan_nanograv import partial_class


STYLE = {"text.usetex": True,
         "font.family": "serif",
         "axes.labelsize": 28,
         "legend.fontsize": 28,
         "xtick.labelsize": 28,
         "ytick.labelsize": 28,
         "xtick.major.size": 8,
         "ytick.major.size": 8,
         "text.latex.preamble": "\\usepackage{txfonts}"}


plt.rcParams.update(**STYLE)


def add_violin(ax, x, y, color, label, rw=0.1, alpha=0.8):

    vpt = ax.violinplot(y, positions=(x), widths=rw * x, showextrema=False)

    for pc in vpt['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(alpha)

    ax.stairs(np.arange(1, 6, 1), fill=True, color=color, alpha=alpha, label=label)


def add_violins(ax):
    add_violin(ax, freqs_NANO12_5, NANO12_logOmegah2_distribution[::5][[0,1,2,3,4]], 'r', 'NANOGrav 12.5yr')
    add_violin(ax, freqs_NANO15_8, NANO15_logOmegah2_distribution, 'limegreen', 'NANOGrav 15yr')
    add_violin(ax, freqs_PPTADR3_8, PPTADR3_logOmegah2_distribution, 'dodgerblue', 'PPTA DR3')
    add_violin(ax, freqs_EPTADR2Full_10, EPTADR2Full_logOmegah2_distribution, 'orange', 'EPTA DR2 Full')


def make_plot(bp):

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$f$ [Hz]')
    ax.set_ylabel(r'$\Omega_{\rm GW} h^2$')

    # Add experimental results

    # add_violins(ax)

    # Add benchmark

    f, fluid, sw, turb, col = np.loadtxt(bp, unpack=True)

    ax.plot(f, sw, 'C0', lw=3,  ls="--", label='Sound waves', alpha=0.65)
    ax.plot(f, turb, 'C2', lw=3,  ls="--", label='Turbulence', alpha=0.65)

    p1 = ax.plot(f, fluid, 'C3', lw=4, alpha=0.8)
    ax.fill_between(f, y1=1e-3 * fluid, y2=1e3 * fluid, color='C3', lw=4, alpha=0.2)
    p2 = ax.fill(np.NaN, np.NaN, 'C3', alpha=0.2)

    ax.plot(f, col, 'grey', lw=3, ls=":", label='Collisions')

    p2[0].set(lw=4, edgecolor="C3")
    handles, labels = ax.get_legend_handles_labels()
    handles.append((p2[0], p1[0]))

    labels.append('Fluid --- sound waves + turbulence')
    ax.legend(handles, labels)

    return ax


def write_gw_predictions(folder_name, file_name):
    potential = partial_class(SMplusCubic, bUseBoltzmannSuppression=True)
    gw1 = GWAnalyser(LISA, potential, folder_name, bForceAllTransitionsRelevant=False)
    gw1.determineGWs_withColl(file_name=file_name)


if __name__ == "__main__":

    write_gw_predictions("BP1/", "bp1_gws.txt")
    ax = make_plot("bp1_gws.txt")
    ax.set_xlim(1e-10, 1e2)
    ax.set_ylim(1e-16, 1e-4)
    plt.tight_layout()
    plt.savefig("GWs_BP1_PTA.pdf")

    write_gw_predictions("BP2/", "bp2_gws.txt")
    ax = make_plot("bp2_gws.txt")
    ax.set_xlim(1e-10, 1e2)
    ax.set_ylim(1e-16, 1e1)
    plt.tight_layout()
    plt.savefig("GWs_BP2_PTA.pdf")
