"""
Generate fig from our nanograv paper from benchmarks on disk
============================================================
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, interpolate

from analysis import phase_structure
from gws import hydrodynamics
from models.Archil_model import SMplusCubic


STYLE = {"text.usetex": True,
         "font.family": "serif",
         "axes.labelsize": 40,
         "legend.fontsize": 28,
         "xtick.labelsize": 28,
         "ytick.labelsize": 28,
         "xtick.major.size": 8,
         "ytick.major.size": 8,
         "text.latex.preamble": "\\usepackage{txfonts}"}


plt.rcParams.update(**STYLE)


def get_report(output_folder: str):
    with open(os.path.join(output_folder, 'phase_history.json'), 'r',  encoding='utf8') as ph:
        return json.load(ph)

def make_pf_plot(bp1, bp2):

    reports = [get_report(bp1) , get_report(bp2)]

    T = [r['transitions'][0]['TSubsample'] for r in reports]
    Pf = [r['transitions'][0]['Pf'] for r in reports]
    Tn = [r['transitions'][0].get('Tn') for r in reports]
    Tp = [r['transitions'][0].get('Tp') for r in reports]
    Tf = [r['transitions'][0].get('Tf') for r in reports]
    names = ["BP1", "BP2"]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.figure(figsize=(12, 8))

    # tweak axes to show lines clearly as suggested by ref

    ax = plt.gca()
    ax.spines['bottom'].set_bounds(0, 70)
    ax.spines['left'].set_bounds(0, 1)
    ax.spines[['right', 'top']].set_visible(False)
    ymin = -0.04
    xmin = -1.5

    for i in range(2):

        # plot curves

        plt.plot(T[i], Pf[i], linewidth=2.5, c=colors[i], label=names[i], zorder=10)

    for i in range(2):

        # plot nucleation temperatures

        if Tn[i] is not None:
            plt.vlines(x=Tn[i], ymin=ymin, ymax=0.99, lw=2, color=colors[i], ls=':', label='$T_n$' if i == 0 else "_nolegend_")

        # plot percolation temperatures

        if Tp[i] is not None:
            critical_pf = 0.71
            plt.vlines(x=Tp[i], ymin=ymin, ymax=critical_pf, lw=2, color=colors[i], ls='--', label='$T_p$' if i == 0 else "_nolegend_")
            plt.hlines(y=critical_pf, xmin=xmin, xmax=Tp[i], lw=2, color=colors[i], ls='--')

        # plot completion temperatures

        if Tf[i] is not None:
            critical_pf = 0.01
            plt.vlines(x=Tf[i], ymin=ymin, ymax=critical_pf, lw=2, color=colors[i], ls='-.', label='$T_f$' if i == 0 else "_nolegend_")
            plt.hlines(y=critical_pf, xmin=xmin, xmax=Tf[i], lw=2, color=colors[i], ls='-.')

    # make legend

    leg = plt.legend(loc="lower right")

    # adjust colors on legend

    for i in range(2, 5):
        leg.legend_handles[i].set_color('black')

    # finalize

    plt.xlabel(r'$T$ [GeV]')
    plt.ylabel('$P_{f}(T)$')

    plt.xlim(xmin, 70)
    plt.ylim(ymin, 1.01)

    plt.tight_layout()
    plt.subplots_adjust(right=0.98)

    # save to disk

    folder_name = os.path.join("output", "nanograv")
    os.makedirs(folder_name, exist_ok=True)
    plt.savefig(os.path.join(folder_name, "Pf_combined.pdf"))


def reheat_temp(T, Tc, rho_f, rho_t_func):
    def objective(t):
        return rho_t_func(t) - rho_f

    maxT = Tc

    # handle cases where reheating takes us past Tc
    while objective(maxT) < 0:
        maxT *= 2

    return optimize.toms748(objective, T, maxT)


def make_t_reh_plot(folder_name):

    # load from disk

    report = get_report(folder_name)
    success, structure = phase_structure.load_data(os.path.join(folder_name, "phase_structure.dat"))
    potential = SMplusCubic(*np.loadtxt(os.path.join(folder_name, "parameter_point.txt")))

    if not success:
        raise RuntimeError('Failed to load phase structure.')

    # generate T_reh

    transition_report = report['transitions'][0]
    from_phase = structure.phases[transition_report['falsePhase']]
    to_phase = structure.phases[transition_report['truePhase']]

    Tc = transition_report['Tc']
    T = transition_report['TSubsample']

    energy_T = np.linspace(min(T), Tc, 200)
    res = [hydrodynamics.calculateEnergyDensityAtT(from_phase, to_phase, potential, t) for t in energy_T]
    rhof, rhot = zip(*res)
    rhof_interp = interpolate.CubicSpline(energy_T, rhof)
    rhot_interp = interpolate.CubicSpline(energy_T, rhot)

    Treh = [reheat_temp(t, Tc, rhof_interp(t), rhot_interp) for t in T]

    # plot data

    plt.figure(figsize=(12, 8))

    plt.plot(T, Treh, lw=2.5)
    plt.plot(T, T, lw=1.75, ls='--')

    # finalize

    plt.xlim(0, None)
    plt.ylim(0, None)
    plt.xlabel(r'$T_p$ [GeV]')
    plt.ylabel(r'$T_{\mathrm{reh}}$ [GeV]')
    plt.tight_layout()

    folder_name = os.path.join("output", "nanograv")
    os.makedirs(folder_name, exist_ok=True)
    plt.savefig(os.path.join(folder_name, "Treh_vs_Tp.pdf"))



if __name__ == "__main__":
    make_pf_plot("BP1", "BP2")
    make_t_reh_plot("BP2")
