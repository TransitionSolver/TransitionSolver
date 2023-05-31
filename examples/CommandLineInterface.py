from __future__ import annotations
import traceback
from models.ToyModel import ToyModel
from models.RealScalarSingletModel import RealScalarSingletModel
from models.RealScalarSingletModel_HT import RealScalarSingletModel_HT
from models.SingletModel import SingletModel
from analysis import PhaseStructure, PhaseHistoryAnalysis, TransitionGraph
import numpy as np
import subprocess
import json
import time
import pathlib


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


def main(potentialClass, outputFolder, PT_script, PT_params, parameterPoint):
    # Create the output folder if it doesn't exist already.
    pathlib.Path(str(pathlib.Path(outputFolder))).mkdir(parents=True, exist_ok=True)

    # Save the parameter point for future reference (e.g. so we know what parameter point resulted in the reported phase
    # history). Also, we can have PhaseTracer construct the potential by reading the saved parameter point.
    np.savetxt(outputFolder + '/parameter_point.txt', np.array([parameterPoint]))

    config = None
    PhaseTracer_directory = ''
    windows = False

    # Load the relative path to PhaseTracer from the config file.
    try:
        with open('config/config_user.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError as e:
        traceback.print_exc()
        print('Unable to load configuration file.')
        sys.exit(1)

    try:
        PhaseTracer_directory = config['PhaseTracer_directory']
    except KeyError:
        print('Unable to load PhaseTracer directory from the configuration file.')

    try:
        windows = config['Windows']
    except KeyError:
        pass  # Just assume not Windows.

    if PhaseTracer_directory == '':
        sys.exit(1)

    # Call PhaseTracer to determine the phase structure of the potential. 'wsl' is the Windows Subsystem for Linux,
    # which is required because PhaseTracer does not run on Windows directly. The second element of the list is the
    # program name. The remaining elements are the input parameters for the specified program. The timeout (in seconds)
    # ensures that PhaseTracer cannot run indefinitely. stdout is routed to DEVNULL to suppress any print statements
    # from PhaseTracer. stderr is routed to STDOUT so that errors in PhaseTracer are printed here.
    command = (['wsl'] if windows else []) + [PhaseTracer_directory + f'bin/{PT_script}', outputFolder +
        '/parameter_point.txt', outputFolder] + PT_params
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
    potential = potentialClass(*parameterPoint)

    # Analyse the phase history and track the execution time. The timing will be handled internally in a future version
    # of the code.
    startTime = time.perf_counter()
    paths, _ = analyser.analysePhaseHistory_supplied(potential, phaseStructure, vw=1.)
    endTime = time.perf_counter()

    # Write the phase history report. Again, this will be handled within PhaseHistoryAnalysis in a future version of the
    # code.
    writePhaseHistoryReport(outputFolder + '/phase_history.json', paths, phaseStructure, endTime - startTime)


if __name__ == "__main__":
    import sys

    print(sys.argv)

    # Check that the user has included enough parameters in the run command.
    if len(sys.argv) < 4:
        print('Please specify a model, an output folder, and a parameter point.')
        sys.exit(1)

    modelLabel = sys.argv[1].lower()

    # Support model labels.
    modelLabels = ['toy', 'rss', 'rss_ht', 'singlet']
    # The AnalysablePotential subclass corresponding to a particular model label.
    models = [ToyModel, RealScalarSingletModel, RealScalarSingletModel_HT, SingletModel]
    # PhaseTracer script to run, specific to a particular model label.
    PT_scripts = ['run_ToyModel', 'run_RSS', 'run_RSS','run_SingletModel']
    # Extra arguments to pass to PhaseTracer, specific to a particular model label.
    PT_paramArrays = [[], [], ['-ht'], []]
    _potentialClass = None
    _PT_script = ''
    _PT_params = []

    # Attempt to match the input model label to the supported model labels.
    for i in range(len(models)):
        if modelLabel == modelLabels[i]:
            _potentialClass = models[i]
            _PT_script = PT_scripts[i]
            _PT_params = PT_paramArrays[i]
            break

    if _potentialClass is None:
        print(f'Invalid model label: {modelLabel}. Valid model labels are: {modelLabels}')
        sys.exit(1)

    outputFolder = sys.argv[2]

    _parameterPoint = sys.argv[3]
    loadedParameterPoint = False

    # First, attempt to treat the parameter point as a file name.
    if len(_parameterPoint) > 3:
        if _parameterPoint[-4:] == '.txt':
            try:
                with open(_parameterPoint, 'r') as f:
                    _parameterPoint = np.loadtxt(_parameterPoint)
                    loadedParameterPoint = True
            except:
                pass

    # If the parameter point does not correspond to the name of a readable file, treat it as a list of parameter values.
    if not loadedParameterPoint:
        try:
            _parameterPoint = []
            for i in range(3, len(sys.argv)):
                _value = float(sys.argv[i])
                _parameterPoint.append(_value)
        except:
            print('Failed to load parameter point defined by:', ' '.join(sys.argv[2:]))
            sys.exit(1)

    main(_potentialClass, outputFolder, _PT_script, _PT_params, _parameterPoint)
