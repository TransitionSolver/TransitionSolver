from __future__ import annotations
from models.ToyModel import ToyModel
from models.SingletModel import SingletModel
from models.RealScalarSingletModel import RealScalarSingletModel
from analysis import PhaseStructure, PhaseHistoryAnalysis, TransitionGraph
import numpy as np
import subprocess
import json
import time
import pathlib


# The file path to PhaseTracer. This is user specific.
#PHASETRACER_DIR = '/home/xuzhongxiu/PhaseTracer/'

import traceback
import sys



def writePhaseHistoryReport(fileName: str, paths: list[TransitionGraph.ProperPath], phaseStructure:
        PhaseStructure.PhaseStructure, analysisTime: float):
    report = {}

    if len(phaseStructure.transitions) > 0:
        report['transitions'] = [t.getReport(fileName) for t in phaseStructure.transitions]
    if len(paths) > 0:
        report['paths'] = [p.getReport() for p in paths]
    report['valid'] = any([p.bValid for p in paths])
    report['analysisTime'] = analysisTime

    print('Writing report...')

    try:
        with open(fileName, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
    except (json.decoder.JSONDecodeError, TypeError):
        print('We have a JSON serialisation error. The report is:')
        print(report)
        print('Failed to write report')


def main():
    # The folder where we will write the output.
    #outputFolder = 'output/example3'
    outputFolder = 'output/RSS/RSS_BP1'
    # Create the output folder if it doesn't exist already.
    pathlib.Path(str(pathlib.Path(outputFolder))).mkdir(parents=True, exist_ok=True)

    # Define a parameter point (i.e. values for each parameter in the model).
    #parameterPoint = [0.104005, 250, 3.5, 0.2]
    parameterPoint = np.loadtxt('input/RSS/RSS_BP1.txt')

    # Save the parameter point for future reference (e.g. so we know what parameter point resulted in the reported phase
    # history). Also, we can have PhaseTracer construct the potential by reading the saved parameter point.
    np.savetxt(outputFolder + '/parameter_point.txt', np.array([parameterPoint]))

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

    # Call PhaseTracer to determine the phase structure of the potential. 'wsl' is the Windows Subsystem for Linux,
    # which is required because PhaseTracer does not run on Windows directly. The second element of the list is the
    # program name. The remaining elements are the input parameters for the specified program. The timeout (in seconds)
    # ensures that PhaseTracer cannot run indefinitely. shell=True is required so that WSL can be called from the shell.
    # stdout is routed to DEVNULL to suppress any print statements from PhaseTracer. stderr is routed to STDOUT so that
    # errors in PhaseTracer are printed here.


#subprocess.call([ PHASETRACER_DIR + 'bin/run_ToyModel', outputFolder + '/parameter_point.txt', outputFolder],timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
   # subprocess.call([ PHASETRACER_DIR + 'bin/run_SingletModel', outputFolder + '/parameter_point.txt', outputFolder],
        timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
   # subprocess.call([ PHASETRACER_DIR + 'bin/run_RSS', outputFolder + '/parameter_point.txt', outputFolder],
   #    timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


    command = (['wsl'] if windows else []) + [PhaseTracer_directory + 'bin/run_RSS', outputFolder +
        '/parameter_point.txt', outputFolder]
    subprocess.call(command, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


    # Load the phase structure saved by PhaseTracer.
    bFileExists, phaseStructure = PhaseStructure.load_data(outputFolder + '/phase_structure.dat', bExpectFile=True)

    # Validate the phase structure.
    if not bFileExists:
        print('Could not find phase structure file.')
        return
    if len(phaseStructure.transitionPaths) == 0:
        print('No valid transition path to the current phase of the Universe.')
        return

    # Load and configure a PhaseHistoryAnalyser object.
    analyser = PhaseHistoryAnalysis.PhaseHistoryAnalyser()
    analyser.bDebug = True
    analyser.bPlot = True
    analyser.bReportAnalysis = True
    analyser.bReportPaths = True
    analyser.timeout_phaseHistoryAnalysis = 100

    # Create the potential using the parameter point.
    #potential = ToyModel(*parameterPoint)
    #potential = SingletModel(*parameterPoint)
    potential = RealScalarSingletModel(*parameterPoint)

    # Analyse the phase history and track the execution time. The timing will be handled internally in a future version
    # of the code.
    startTime = time.perf_counter()
    paths, _ = analyser.analysePhaseHistory_supplied(potential, phaseStructure, vw=1.)
    endTime = time.perf_counter()

    # Write the phase history report. Again, this will be handled within PhaseHistoryAnalysis in a future version of the
    # code.
    writePhaseHistoryReport(outputFolder + '/phase_history.json', paths, phaseStructure, endTime - startTime)


if __name__ == "__main__":
    main()
