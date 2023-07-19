import pathlib
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import json

from scipy import optimize, interpolate

import command_line_interface as cli
from analysis import phase_structure
from gws import hydrodynamics

from models.Archil_model import SMplusCubic


def runPoint(inputFileName: str, outputFolderName: str) -> None:
    parameterPoint = list(np.loadtxt(inputFileName))
    cli.main(SMplusCubic, outputFolderName, 'run_supercool', ['-boltz', '-debug'], parameterPoint, bDebug=True,
        bUseBoltzmannSuppression=True)


def getReport(inputFileName: str, outputFolderName: str, generateIfNotExist: bool = True):
    report = None
    try:
        with open(outputFolderName+'/phase_history.json', 'r') as f:
            report = json.load(f)
    except FileNotFoundError:
        if generateIfNotExist:
            print('Output does not exist, attempting to generate...')
            runPoint(inputFileName, outputFolderName)
            report = getReport(inputFileName, outputFolderName, generateIfNotExist=False)
            print('Finished generating output.')
        else:
            print('Output does not exist!')

    return report


def makePfPlot():
    BP = 1
    reportBP1 = getReport(f'input/nanograv/nanograv_BP{BP}.txt', f'output/nanograv/BP{BP}')
    BP = 2
    reportBP2 = getReport(f'input/nanograv/nanograv_BP{BP}.txt', f'output/nanograv/BP{BP}')

    if reportBP1 is None:
        print('Phase history report was not obtained successfully for BP1, cannot generate plots.')

    if reportBP2 is None:
        print('Phase history report was not obtained successfully for BP2, cannot generate plots.')

    T1 = reportBP1['transitions'][0]['TSubsample']
    T2 = reportBP2['transitions'][0]['TSubsample']
    Pf1 = reportBP1['transitions'][0]['Pf']
    Pf2 = reportBP2['transitions'][0]['Pf']

    Tn1 = reportBP1['transitions'][0].get('Tn', -1)
    Tn2 = reportBP2['transitions'][0].get('Tn', -1)
    Tp1 = reportBP1['transitions'][0].get('Tp', -1)
    Tp2 = reportBP2['transitions'][0].get('Tp', -1)
    Tf1 = reportBP1['transitions'][0].get('Tf', -1)
    Tf2 = reportBP2['transitions'][0].get('Tf', -1)

    plt.rcParams.update({"text.usetex": True})
    colourCycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    legendHandles = []
    plt.figure(figsize=(12, 8))
    line = plt.plot(T1, Pf1, linewidth=2.5, zorder=(1 if T1[0] > T2[0] else 0), c=colourCycle[0])
    legendHandles.append(line)
    line = plt.plot(T2, Pf2, linewidth=2.5, zorder=(1 if T2[0] > T1[0] else 0), c=colourCycle[1])
    legendHandles.append(line)
    if Tn1 > 0: plt.axvline(Tn1, ymax=1., lw=2, c=colourCycle[0], ls=':', label='$T_n$')
    if Tn2 > 0: plt.axvline(Tn2, ymax=1, lw=2, c=colourCycle[1], ls=':', label='_nolegend_')
    if Tp1 > 0:
        plt.axvline(Tp1, ymax=0.71, lw=2, c=colourCycle[0], ls='--', label='$T_p$')
        plt.axhline(0.71, xmin=0, xmax=Tp1/max(T1[0], T2[0]), lw=2, c=colourCycle[0], ls='--', label='_nolegend_',
            zorder=(1 if Tp2 > Tp1 else 0))
    if Tp2 > 0:
        plt.axvline(Tp2, ymax=0.71, lw=2, c=colourCycle[1], ls='--', label='_nolegend_')
        plt.axhline(0.71, xmin=0, xmax=Tp2/max(T1[0], T2[0]), lw=2, c=colourCycle[1], ls='--', label='_nolegend_',
            zorder=(1 if Tp1 > Tp2 else 0))
    if Tf1 > 0:
        plt.axvline(Tf1, ymax=0.01, lw=2, c=colourCycle[0], ls='-.', label='$T_f$')
        plt.axhline(0.01, xmin=0, xmax=Tf1/max(T1[0], T2[0]), lw=2, c=colourCycle[0], label='_nolegend_', ls='-.')
    if Tf2 > 0:
        plt.axvline(Tf2, ymax=0.01, lw=2, c=colourCycle[1], ls='-.', label='_nolegend_')
        plt.axhline(0.01, xmin=0, xmax=Tf2/max(T1[0], T2[0]), lw=2, c=colourCycle[1], label='_nolegend_', ls='-.')
    plt.xlabel('$T \, \\mathrm{[GeV]}$', fontsize=40)
    plt.ylabel('$P_{\\! f}(T)$', fontsize=40)
    plt.tick_params(size=8, labelsize=28)
    plt.xlim(left=0, right=max(T1[0], T2[0]))
    plt.ylim(bottom=0, top=1)
    legendLabels = ['$\\mathrm{BP1}$', '$\\mathrm{BP2}$', '$T_n$', '$T_p$', '$T_f$']
    plt.legend(legendLabels, fontsize=32)
    legend = plt.gca().get_legend()
    for i in range(2, 5):
        legend.legendHandles[i].set_color('black')
    #plt.legend(handles=legendHandles, labels=legendLabels, fontsize=32)
    plt.margins(0, 0)
    plt.tight_layout()
    #plt.show()
    pathlib.Path(str(pathlib.Path('output/nanograv'))).mkdir(parents=True, exist_ok=True)
    plt.savefig("output/nanograv/Pf_combined.pdf")


# Copied from transition_analysis.TransitionAnalyer.calculateReheatTemperature.
def calculateReheatingTemperature(T: float, Tc: float, rho_f: float, rho_t_func: Callable[[float], float]) -> float:
    def objective(t):
        rho_t = rho_t_func(t)
        # Conservation of energy => rhof = rhof*Pf + rhot*Pt which is equivalent to rhof = rhot (evaluated at
        # different temperatures, T and Tt (Treh), respectively).
        return rho_t - rho_f

    maxT = Tc

    # Handle cases where reheating takes us past Tc.
    while objective(maxT) < 0:
        maxT *= 2

    return optimize.toms748(objective, T, maxT)


def makeTrehPlot():
    BP = 2
    reportBP2 = getReport(f'input/nanograv/nanograv_BP{BP}.txt', f'output/nanograv/BP{BP}/')

    if reportBP2 is None:
        print('Phase history report was not obtained successfully for BP2, cannot generate plots.')

    bSuccess, phaseStructure = phase_structure.load_data(f'output/nanograv/BP{BP}/phase_structure.dat')

    if not bSuccess:
        print('Failed to load phase structure.')
        return

    transitionReport = reportBP2['transitions'][0]

    fromPhase = phaseStructure.phases[transitionReport['falsePhase']]
    toPhase = phaseStructure.phases[transitionReport['truePhase']]
    Tc = transitionReport['Tc']
    potential = SMplusCubic(*np.loadtxt(f'output/nanograv/BP{BP}/parameter_point.txt'))

    allT: List[float] = transitionReport['TSubsample']
    energy_T: np.ndarray = np.linspace(allT[-1], transitionReport['Tc'], 200)

    rhof: List[float] = []
    rhot: List[float] = []
    Treh: List[float] = []

    for t in energy_T:
        ef, et = hydrodynamics.calculateEnergyDensityAtT(fromPhase, toPhase, potential, t)
        rhof.append(ef)
        rhot.append(et)

    rhof_interp = interpolate.CubicSpline(energy_T, rhof)
    rhot_interp = interpolate.CubicSpline(energy_T, rhot)

    for t in allT:
        Treh.append(calculateReheatingTemperature(t, Tc, rhof_interp(t), rhot_interp))

    plt.rcParams.update({"text.usetex": True})

    plt.figure(figsize=(12, 8))
    plt.plot(allT, Treh, lw=2.5)
    plt.plot(allT, allT, lw=1.75, ls='--')
    plt.margins(0, 0)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel('$T_p \, \\mathrm{[GeV]}$', fontsize=40)
    plt.ylabel('$T_{\\mathrm{reh}} \, \\mathrm{[GeV]}$', fontsize=40)
    plt.tick_params(size=8, labelsize=28)
    plt.tight_layout()
    #plt.show()
    pathlib.Path(str(pathlib.Path('output/nanograv'))).mkdir(parents=True, exist_ok=True)
    plt.savefig("output/nanograv/Treh_vs_Tp.pdf")


#Just doubling code because I'm lazy and we're not using this plot anyway.
def makeCombinedTrehPlot():
    BP = 1
    reportBP1 = getReport(f'input/nanograv/nanograv_BP{BP}.txt', f'output/nanograv/BP{BP}/')
    BP = 2
    reportBP2 = getReport(f'input/nanograv/nanograv_BP{BP}.txt', f'output/nanograv/BP{BP}/')

    if reportBP1 is None:
        print('Phase history report was not obtained successfully for BP1, cannot generate plots.')

    if reportBP2 is None:
        print('Phase history report was not obtained successfully for BP2, cannot generate plots.')

    bSuccess, phaseStructure1 = phase_structure.load_data(f'output/nanograv/BP{BP}/phase_structure.dat')

    if not bSuccess:
        print('Failed to load phase structure for BP1.')
        return

    bSuccess, phaseStructure2 = phase_structure.load_data(f'output/nanograv/BP{BP}/phase_structure.dat')

    if not bSuccess:
        print('Failed to load phase structure for BP2.')
        return

    transitionReport1 = reportBP1['transitions'][0]
    transitionReport2 = reportBP2['transitions'][0]

    fromPhase1 = phaseStructure1.phases[transitionReport1['falsePhase']]
    toPhase1 = phaseStructure1.phases[transitionReport1['truePhase']]
    Tc1 = transitionReport1['Tc']
    potential1 = SMplusCubic(*np.loadtxt(f'output/nanograv/BP{BP}/parameter_point.txt'))

    fromPhase2 = phaseStructure2.phases[transitionReport1['falsePhase']]
    toPhase2 = phaseStructure2.phases[transitionReport1['truePhase']]
    Tc2 = transitionReport2['Tc']
    potential2 = SMplusCubic(*np.loadtxt(f'output/nanograv/BP{BP}/parameter_point.txt'))

    allT1: List[float] = transitionReport1['TSubsample']
    allT2: List[float] = transitionReport2['TSubsample']
    energy_T1: np.ndarray = np.linspace(allT1[-1], transitionReport1['Tc'], 200)
    energy_T2: np.ndarray = np.linspace(allT2[-1], transitionReport2['Tc'], 200)

    rhof1: List[float] = []
    rhot1: List[float] = []
    Treh1: List[float] = []
    rhof2: List[float] = []
    rhot2: List[float] = []
    Treh2: List[float] = []

    for t in energy_T1:
        ef, et = hydrodynamics.calculateEnergyDensityAtT(fromPhase1, toPhase1, potential1, t)
        rhof1.append(ef)
        rhot1.append(et)

    for t in energy_T2:
        ef, et = hydrodynamics.calculateEnergyDensityAtT(fromPhase2, toPhase2, potential2, t)
        rhof2.append(ef)
        rhot2.append(et)

    rhof_interp1 = interpolate.CubicSpline(energy_T1, rhof1)
    rhot_interp1 = interpolate.CubicSpline(energy_T1, rhot1)
    rhof_interp2 = interpolate.CubicSpline(energy_T2, rhof2)
    rhot_interp2 = interpolate.CubicSpline(energy_T2, rhot2)

    for t in allT1:
        Treh1.append(calculateReheatingTemperature(t, Tc1, rhof_interp1(t), rhot_interp1))

    for t in allT2:
        Treh2.append(calculateReheatingTemperature(t, Tc2, rhof_interp2(t), rhot_interp2))

    #plt.rcParams.update({"text.usetex": True})

    plt.figure(figsize=(12, 8))
    plt.plot(allT1, Treh1, lw=0.5)
    plt.plot(allT2, Treh2, lw=0.5)
    #plt.plot(allT1, allT1, lw=1.75, ls='--')
    plt.margins(0, 0)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel('$T_p \, \\mathrm{[GeV]}$', fontsize=40)
    plt.ylabel('$T_{\\mathrm{reh}} \, \\mathrm{[GeV]}$', fontsize=40)
    plt.tick_params(size=8, labelsize=28)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    #makePfPlot()
    makeTrehPlot()
