from __future__ import annotations
from src.TransitionSolver.analysis.transition_analysis import TransitionAnalyser
from src.TransitionSolver.models.toy_model import ToyModel
from src.TransitionSolver.models.real_scalar_singlet_model_boltz import RealScalarSingletModel_Boltz
from src.TransitionSolver.analysis.phase_structure import PhaseStructure
from src.TransitionSolver.analysis.phase_history_analysis import PhaseHistoryAnalyser, AnalysisMetrics
from src.TransitionSolver.analysis.transition_graph import Path
from src.TransitionSolver.analysis import phase_structure
import numpy as np
import subprocess
import json
import pathlib
import traceback
import sys
from src.TransitionSolver.util.events import notifyHandler


def writePhaseHistoryReport(fileName: str, paths: list[Path], phaseStructure: PhaseStructure, analysisMetrics:
        AnalysisMetrics) -> None:
    report = {}

    if len(phaseStructure.transitions) > 0:
        report['transitions'] = [t.getReport(fileName) for t in phaseStructure.transitions]
    if len(paths) > 0:
        report['paths'] = [p.getReport() for p in paths]
    report['valid'] = any([p.is_valid for p in paths])
    report['analysisTime'] = analysisMetrics.analysisElapsedTime

    print('Writing report...')

    try:
        with open(fileName, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
    except (json.decoder.JSONDecodeError, TypeError):
        print('We have a JSON serialisation error. The report is:')
        print(report)
        print('Failed to write report.')


def main():
    # The folder where we will write the output.
    outputFolder = 'output/RSS/RSS_BP1'
    # Create the output folder if it doesn't exist already.
    pathlib.Path(str(pathlib.Path(outputFolder))).mkdir(parents=True, exist_ok=True)

    # Define a parameter point (i.e. values for each parameter in the model).
    # parameterPoint should be a comma separated list of numerical inputs like "parameterPoint = np.array([x,y,z,...])"
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
    command = (['wsl'] if windows else []) + [PhaseTracer_directory + 'bin/run_RSS', outputFolder +
        '/parameter_point.txt', outputFolder] + ['-boltz']
    subprocess.call(command, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print("after calling command = ", command)
    # Load the phase structure saved by PhaseTracer.
    bFileExists, phaseStructure = phase_structure.load_data(outputFolder + '/phase_structure.dat')

    # Validate the phase structure.
    if not bFileExists:
        print('Could not find phase structure file.')
        return
    if len(phaseStructure.transitionPaths) == 0:
        print('No valid transition path to the current phase of the Universe.')
        return

    # Load and configure a PhaseHistoryAnalyser object.
    analyser = PhaseHistoryAnalyser()
    analyser.bDebug = True
    analyser.bPlot = True
    analyser.bReportAnalysis = True
    analyser.bReportPaths = True
    analyser.timeout_phaseHistoryAnalysis = 100

    # Create the potential using the parameter point.
    potential = RealScalarSingletModel_Boltz(*parameterPoint)
    def notify_TransitionAnalyser_on_create(transitionAnalyser: TransitionAnalyser):
        transitionAnalyser.bComputeSubsampledThermalParams = True

    notifyHandler.addEvent('TransitionAnalyser-on_create', notify_TransitionAnalyser_on_create)

    # Analyse the phase history.
    paths, _, analysisMetrics = analyser.analysePhaseHistory_supplied(potential, phaseStructure, vw=0.9)

    # Write the phase history report. Again, this will be handled within PhaseHistoryAnalysis in a future version of the
    # code.
    writePhaseHistoryReport(outputFolder + '/phase_history.json', paths, phaseStructure, analysisMetrics)


if __name__ == "__main__":
    main()
