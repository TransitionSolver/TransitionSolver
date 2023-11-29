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
         "axes.labelsize": 28,
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
    ax.spines[['right', 'top']].set_visible(False)

    ymin = 0
    ymax = 1.
    xmin = -1.
    xmax = 70.

    ax.spines['left'].set_bounds(ymin, ymax)

    for i in range(2):

        # plot curves

        plt.plot(T[i], Pf[i], linewidth=2.5, c=colors[i], label=names[i], zorder=10)

    for i in range(2):

        style = {'color': colors[i], 'ls': '--', 'lw': 2, 'alpha': 0.45}

        # plot nucleation temperatures

        if Tn[i] is not None:
            plt.vlines(x=Tn[i], ymin=ymin, ymax=ymax, **style)

        # plot percolation temperatures

        if Tp[i] is not None:
            critical_pf = 0.71
            plt.vlines(x=Tp[i], ymin=ymin, ymax=critical_pf, **style)
            plt.hlines(y=critical_pf, xmin=xmin, xmax=Tp[i], **style)

        # plot completion temperatures

        if Tf[i] is not None:
            critical_pf = 0.01
            plt.vlines(x=Tf[i], ymin=ymin, ymax=critical_pf, **style)
            plt.hlines(y=critical_pf, xmin=xmin, xmax=Tf[i], **style)

    # annotate

    fontsize = 15

    plt.annotate(r"$P_f(T_p) = 0.71$, $T_p = 0.1\,\textrm{GeV}$", (Tp[1] + 0.5, 0.15), rotation=-90., color=colors[1], fontsize=fontsize)
    plt.annotate(r"Unit nucleation, $T_n = 53\,\textrm{GeV}$", (Tn[0] + 0.5, 0.25), rotation=-90., color=colors[0], fontsize=fontsize)
    plt.annotate(r"$P_f(T_f) = 0.01$, $T_f = 24\,\textrm{GeV}$", (2.5, 0.01 + 0.02), rotation=0., color=colors[0], fontsize=fontsize)
    plt.annotate(r"$P_f(T_p) = 0.71$, $T_p = 37\,\textrm{GeV}$", (9.5, 0.71 + 0.02), rotation=0., color=colors[0], fontsize=fontsize)

    # make legend

    plt.legend(loc="center right", bbox_to_anchor=(1.01, 0.75))

    # finalize

    plt.xlabel(r'$T$ [GeV]')
    plt.ylabel('False vacuum fraction, $P_{f}(T)$')

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax + 0.1)

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

    Tp = [37., 0.1]
    BP = [reheat_temp(t, Tc, rhof_interp(t), rhot_interp) for t in Tp]

    # plot data

    plt.figure(figsize=(12, 8))

    plt.plot(T, Treh, lw=2.5)
    plt.plot(T, T, lw=1.75, ls='--')

    plt.scatter(Tp, BP, marker="H", s=50, clip_on=False, zorder=100)

    # annotate

    plt.annotate("BP1", (Tp[0], BP[0] + 2), ha="center", fontsize=20)
    plt.annotate("BP2", (Tp[1] + 0.5, BP[1] + 1), fontsize=20)


    # finalize

    plt.xlim(0, max(T))
    plt.ylim(0, max(T))
    plt.xlabel(r'$T_p$ [GeV]')
    plt.ylabel(r'$T_{\mathrm{reh}}$ [GeV]')
    plt.tight_layout()

    folder_name = os.path.join("output", "nanograv")
    os.makedirs(folder_name, exist_ok=True)
    plt.savefig(os.path.join(folder_name, "Treh_vs_Tp.pdf"))



if __name__ == "__main__":
    make_pf_plot("BP1", "BP2")
    make_t_reh_plot("BP2")
