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

from analysis import phase_structure
from analysis.phase_history_analysis import PhaseHistoryAnalyser, AnalysisMetrics
from analysis.phase_structure import PhaseStructure
from analysis.transition_analysis import TransitionAnalyser
from analysis.transition_graph import ProperPath
from models.Archil_model import SMplusCubic
from util.events import notifyHandler


def writePhaseHistoryReport(fileName: str, paths: list[ProperPath], phaseStructure: PhaseStructure, analysisMetrics:
        AnalysisMetrics) -> None:
    report = {}

    if len(phaseStructure.transitions) > 0:
        report['transitions'] = [t.getReport(fileName) for t in phaseStructure.transitions]
    if len(paths) > 0:
        report['paths'] = [p.getReport() for p in paths]
    report['valid'] = any([p.bValid for p in paths])
    report['analysisTime'] = analysisMetrics.analysisElapsedTime

    print('Writing report...')

    try:
        with open(fileName, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
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

    potential = SMplusCubic(parameterPointFunction(pointIndex))
    pathlib.Path(outputFolderName).mkdir(parents=True, exist_ok=True)
    #np.savetxt(outputFolderName + '/parameter_point.txt', np.array(potential.getParameterPoint()), delimiter=' ')
    with open(outputFolderName + '/parameter_point.txt', 'w') as f:
        f.write(' '.join([str(param) for param in potential.getParameterPoint()]))
    command = (['wsl'] if windows else []) + [PhaseTracer_directory + f'bin/run_supercool', outputFolderName +
        '/parameter_point.txt', outputFolderName]
    subprocess.call(command, timeout=60)#, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # Load the phase structure saved by PhaseTracer.
    bFileExists, phaseStructure = phase_structure.load_data(outputFolderName + '/phase_structure.dat', bExpectFile=True)

    # Validate the phase structure.
    if not bFileExists:
        print('Could not find phase structure file.')
        return
    if len(phaseStructure.transitionPaths) == 0:
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
    paths, _, analysisMetrics = analyser.analysePhaseHistory_supplied(potential, phaseStructure, vw=1.)

    # Write the phase history report. Again, this will be handled within PhaseHistoryAnalysis in a future version of the
    # code.
    writePhaseHistoryReport(outputFolderName + '/phase_history.json', paths, phaseStructure, analysisMetrics)


def scanWithPipeline(outputFolderName: str, scanIndex: int = 1):
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

    for i in range(numSamples):
        try:
            with open(outputFolderName+f'/{scanIndex}/{i}/phase_history.json', 'r') as f:
                report = json.load(f)

            with open(outputFolderName+f'/{scanIndex}/{i}/parameter_point.txt', 'r') as f:
                kappa.append(float(f.readline().split()[0]))
            Tp.append(report['transitions'][0].get('Tp', 0))
            Tf.append(report['transitions'][0].get('Tf', 0))
            TVphysDecr_high.append(report['transitions'][0].get('TVphysDecr_high', 0))
        except:
            traceback.print_exc()
            continue

    plt.plot(kappa, Tp)
    plt.plot(kappa, Tf)
    plt.plot(kappa, TVphysDecr_high)
    plt.show()


if __name__ == "__main__":
    #scanWithPipeline('output/pipeline/archil', scanIndex=3)
    plotPipeline('output/pipeline/archil', 1, 100)
