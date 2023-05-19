from models.ToyModel import ToyModel
from models.RealScalarSingletModel import RealScalarSingletModel
from models.RealScalarSingletModel_HT import RealScalarSingletModel_HT
from analysis import PhaseStructure, PhaseHistoryAnalysis, TransitionGraph
import numpy as np
import subprocess
import json
import time
import pathlib


# The relative file path to PhaseTracer. This is user specific.
PHASETRACER_DIR = '../../../../../Software/PhaseTracer/'


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


def main(potentialClass, outputFolder, PT_script, PT_params, parameterPoint, windows):
    # Create the output folder if it doesn't exist already.
    pathlib.Path(str(pathlib.Path(outputFolder))).mkdir(parents=True, exist_ok=True)

    # Save the parameter point for future reference (e.g. so we know what parameter point resulted in the reported phase
    # history). Also, we can have PhaseTracer construct the potential by reading the saved parameter point.
    np.savetxt(outputFolder + '/parameter_point.txt', np.array([parameterPoint]))

    # Call PhaseTracer to determine the phase structure of the potential. 'wsl' is the Windows Subsystem for Linux,
    # which is required because PhaseTracer does not run on Windows directly. The second element of the list is the
    # program name. The remaining elements are the input parameters for the specified program. The timeout (in seconds)
    # ensures that PhaseTracer cannot run indefinitely. shell=True is required so that WSL can be called from the shell.
    # stdout is routed to DEVNULL to suppress any print statements from PhaseTracer. stderr is routed to STDOUT so that
    # errors in PhaseTracer are printed here.
    command = (['wsl'] if windows else []) + [PHASETRACER_DIR + f'bin/{PT_script}', outputFolder +
        '/parameter_point.txt', outputFolder] + PT_params
    subprocess.call(command, timeout=60)#, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

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

    # Check that the user has included enough parameters in the run command.
    if len(sys.argv) < 4:
        print('Please specify a model, an output folder, and a parameter point.')
        sys.exit(1)
    # At the moment, Windows users need to prepend the argument 'w' to their run command
    #   (e.g. ./CommandLineInterface.py w <outputFolder> <inputFile>)
    elif len(sys.argv) == 4 and sys.argv[1] == 'w':
        print('Please specify a model, an output folder, and a parameter point.')
        sys.exit(1)

    # The first argument should be 'w' for Windows machines. If 'w' is first, then all other arguments are offset by
    # one position compared to their expected positions.
    if sys.argv[1] == 'w':
        _windows = True
        offset = 1
    else:
        _windows = False
        offset = 0

    modelLabel = sys.argv[1+offset].lower()

    # Support model labels.
    modelLabels = ['toy', 'rss', 'rss_ht']
    # The AnalysablePotential subclass corresponding to a particular model label.
    models = [ToyModel, RealScalarSingletModel, RealScalarSingletModel_HT]
    # PhaseTracer script to run, specific to a particular model label.
    PT_scripts = ['run_ToyModel', 'run_RSS', 'run_RSS']
    # Extra arguments to pass to PhaseTracer, specific to a particular model label.
    PT_paramArrays = [[], [], ['-ht']]
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

    outputFolder = sys.argv[2+offset]

    _parameterPoint = sys.argv[3+offset]
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
            for i in range(3+offset, len(sys.argv)):
                _value = float(sys.argv[i])
                _parameterPoint.append(_value)
        except:
            print('Failed to load parameter point defined by:', ' '.join(sys.argv[2+offset:]))
            sys.exit(1)

    main(_potentialClass, outputFolder, _PT_script, _PT_params, _parameterPoint, _windows)
