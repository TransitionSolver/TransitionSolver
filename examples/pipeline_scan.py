from __future__ import annotations
import json
import pathlib
import subprocess
import sys
import time
import traceback
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

from pathos.multiprocessing import ProcessingPool as Pool




from src.TransitionSolver.models.supercool_model import SMplusCubic
from src.TransitionSolver.analysis.phase_structure import PhaseStructure
from src.TransitionSolver.analysis.phase_history_analysis import PhaseHistoryAnalyser
from src.TransitionSolver.analysis.transition_graph import Path
from src.TransitionSolver.analysis.transition_analysis import TransitionAnalyser, ActionSampler
from src.TransitionSolver.analysis import phase_structure
from src.TransitionSolver.models.analysable_potential import AnalysablePotential
from src.TransitionSolver.util.events import notifyHandler
from TransitionSolver import read_phase_tracer


def writePhaseHistoryReport(fileName: str, paths: list[Path], phaseStructure: PhaseStructure) -> None:
    report = {}

    if len(phaseStructure.transitions) > 0:
        report['transitions'] = [t.report() for t in phaseStructure.transitions]
    if len(paths) > 0:
        report['paths'] = [p.report() for p in paths]
    report['valid'] = any([p.is_valid for p in paths])
    # report['analysisTime'] = analysisMetrics.analysisElapsedTime

    print('Writing report...')

    try:
        with open(fileName, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4, default=str)
    except (json.decoder.JSONDecodeError, TypeError):
        print('We have a JSON serialisation error. The report is:')
        print(report)
        print('Failed to write report.')


def getPipelineParameterPoint(pointIndex: int, numSamples: int, scanIndex: int):
    if scanIndex == 1:
        low = -1.9*125**2/246
        high = -1.8*125**2/246
    elif scanIndex == 2:
        low = -117.8
        high = -116.8
    elif scanIndex == 3:
        low = -117.49
        high = -117.475
    # Narrowed in on Tp -> 0.
    elif scanIndex == 4:
        low = -117.526
        high = -117.516
    # Narrowing in on d/dt Vphys(Tp) -> 0.
    elif scanIndex == 5:
        low = -116.825
        high = -116.795
    # Boltzmann educated guess from non-Boltzmann.
    elif scanIndex == 6:
        low = -117.8
        high = -116.8
    # Boltzmann narrowing in on Tp -> 0.
    elif scanIndex == 7:
        low = -117.561
        high = -117.555
    else:
        low = -2.5
        high = 0
    return low + (high - low)*pointIndex/(numSamples-1)


def pipeline_workerProcess(pointIndex: int, numSamples: int, scanIndex: int, outputFolderName:
        str, parameterPointFunction: Callable[[int], float]):
    outputFolderName += f'/{scanIndex}/{pointIndex}'

    print('Index:', pointIndex)

    # Load the relative path to PhaseTracer from the config file.
    try:
        with open('config/config_user.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        traceback.print_exc()
        print('Unable to load configuration file.')
        sys.exit(1)

    try:
        PhaseTracer_directory = config['PhaseTracer_directory']
    except KeyError:
        print('Unable to load PhaseTracer directory from the configuration file.')
        sys.exit(1)

    try:
        windows = config['Windows']
    except KeyError:
        windows = False  # Just assume not Windows.

    if PhaseTracer_directory == '':
        sys.exit(1)

    potential = SMplusCubic(parameterPointFunction(pointIndex), bUseBoltzmannSuppression=True)
    pathlib.Path(outputFolderName).mkdir(parents=True, exist_ok=True)
    #np.savetxt(outputFolderName + '/parameter_point.txt', np.array(potential.getParameterPoint()), delimiter=' ')
    with open(outputFolderName + '/parameter_point.txt', 'w') as f:
        f.write(' '.join([str(param) for param in potential.getParameterPoint()]))
    command = (['wsl'] if windows else []) + [PhaseTracer_directory + f'bin/run_supercool', outputFolderName +
        '/parameter_point.txt', outputFolderName]
    subprocess.call(command, timeout=60)#, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # Load the phase structure saved by PhaseTracer.
    phaseStructure = phase_structure.read_phase_tracer(phase_tracer_file=outputFolderName + '/phase_structure.dat')

    if len(phaseStructure.paths) == 0:
        print('No valid transition path to the current phase of the Universe.')
        return

    # Load and configure a PhaseHistoryAnalyser object.
    analyser = PhaseHistoryAnalyser()
    analyser.bDebug = False
    analyser.bPlot = False
    analyser.bReportAnalysis = True
    analyser.bReportPaths = True
    analyser.timeout_phaseHistoryAnalysis = 100

    def notify_TransitionAnalyser_on_create(transitionAnalyser: TransitionAnalyser):
        transitionAnalyser.bComputeSubsampledThermalParams = False

    notifyHandler.addEvent('TransitionAnalyser-on_create', notify_TransitionAnalyser_on_create)

    # Analyse the phase history.
    paths, _, _ = analyser.analysePhaseHistory_supplied(potential, phaseStructure, vw=1.)

    # Write the phase history report. Again, this will be handled within PhaseHistoryAnalysis in a future version of the
    # code.
    writePhaseHistoryReport(outputFolderName + '/phase_history.json', paths, phaseStructure)


def scanWithPipeline(outputFolderName: str, scanIndex: int):
    numSamples = 100
    def parameterPointFunction(index: int):
        return getPipelineParameterPoint(index, numSamples,  scanIndex)

    args1 = [i for i in range(numSamples)]
    args2 = [numSamples]*numSamples
    args3 = [scanIndex]*numSamples
    args4 = [outputFolderName]*numSamples
    args5 = [parameterPointFunction]*numSamples
    startTime = time.perf_counter()

    with Pool(nodes=22) as pool:
        results = pool.map(pipeline_workerProcess, args1, args2, args3, args4, args5)

    endTime = time.perf_counter()

    print('\n\nElapsed time:', endTime - startTime)


def plotPipeline(outputFolderName: str, scanIndex: int, numSamples: int):
    kappa = []
    Tp = []
    Tf = []
    TVphysDecr_high = []
    meanBubSep = []

    for i in range(numSamples):
        try:
            with open(outputFolderName+f'/{scanIndex}/{i}/phase_history.json', 'r') as f:
                report = json.load(f)

            with open(outputFolderName+f'/{scanIndex}/{i}/parameter_point.txt', 'r') as f:
                kappa.append(float(f.readline().split()[0]))
            Tp.append(report['transitions'][0].get('Tp', 0))
            Tf.append(report['transitions'][0].get('Tf', 0))
            TVphysDecr_high.append(report['transitions'][0].get('TVphysDecr_high', 0))
            meanBubSep.append(report['transitions'][0].get('meanBubbleSeparation', 0))
        except:
            traceback.print_exc()
            continue

    Tp = np.array(Tp)
    meanBubSep = np.array(meanBubSep)

    plt.figure(figsize=(12, 8))
    plt.plot(kappa, Tp, lw=2.5, marker='.')
    plt.plot(kappa, Tf, lw=2.5, marker='.')
    plt.plot(kappa, TVphysDecr_high, lw=2.5, marker='.')
    plt.legend(['$T_p$', '$T_f$', '$T_d$'])
    plt.xlabel('$\\kappa \,\, \\mathrm{[GeV]}$', fontsize=40)
    plt.ylabel('$T \,\, \\mathrm{[GeV]}$', fontsize=40)
    plt.tick_params(size=10, labelsize=24)
    plt.margins(0, 0)
    plt.tight_layout()
    plt.show()

    #xdat = np.linspace(1, 10, 100)
    #ydat = np.array([1/x+0.001*x for x in xdat])

    from scipy.optimize import curve_fit
    #def recip(x, a, b, c):
    #    return a/(b*x*x*x + c)
    #popt, pcov = curve_fit(recip, Tp[1:], meanBubSep[1:])
    #popt, pcov = curve_fit(recip, xdat, ydat)

    #plt.plot(xdat, ydat, marker='.')
    #plt.plot(xdat, recip(xdat, *popt), marker='.')
    #plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(Tp, meanBubSep, lw=2.5, marker='.')
    #plt.plot(Tp, recip(Tp, 1e18, 1, 0), lw=2.5, marker='.')
    plt.xlabel('$T_p \,\, \\mathrm{[GeV]}$', fontsize=40)
    plt.ylabel('$R_* \,\, \\mathrm{[GeV]}$', fontsize=40)
    plt.tick_params(size=10, labelsize=24)
    plt.margins(0, 0)
    plt.tight_layout()
    plt.show()

    plt.figure()


def debugScanPoint(outputFolderName: str, scanIndex: int, pointIndex: int):
    import command_line_interface as cli
    parameterPoint = list(np.loadtxt(f'{outputFolderName}/{scanIndex}/{pointIndex}/parameter_point.txt'))
    cli.main(SMplusCubic, outputFolderName, 'run_supercool', [], parameterPoint, bDebug=True, bPlot=True,
        bUseBoltzmannSuppression=True)


def checkPotentialAtZeroT():
    potential = SMplusCubic(*np.loadtxt('output/pipeline/SMplusCubicBoltz/7/32/parameter_point.txt'), bUseBoltzmannSuppression=True)
    x = np.linspace(-10, 260, 1000)
    V = potential.Vtot([[X] for X in x], 0., include_radiation=False)
    plt.plot(x, V)
    plt.show()


if __name__ == "__main__":
    #scanWithPipeline('output/pipeline/SMplusCubicBoltz', scanIndex=7)
    plotPipeline('output/pipeline/SMplusCubicBoltz', 6, 100)
    #debugScanPoint('output/pipeline/SMplusCubic', 3, 0)
