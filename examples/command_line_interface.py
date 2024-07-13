from __future__ import annotations
import traceback

from analysis.transition_analysis import TransitionAnalyser
from models.Archil_model import SMplusCubic
from models.analysable_potential import AnalysablePotential
from models.toy_model import ToyModel
from models.real_scalar_singlet_model_boltz import RealScalarSingletModel_Boltz
from models.real_scalar_singlet_model_ht import RealScalarSingletModel_HT
from analysis.phase_structure import PhaseStructure
from analysis.phase_history_analysis import AnalysisMetrics, PhaseHistoryAnalyser
from analysis.transition_graph import ProperPath
from analysis import phase_structure
from typing import Type
import numpy as np
import subprocess
import json
import pathlib
import sys

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


def main(potentialClass: Type[AnalysablePotential], outputFolder: str, PT_script: str, PT_params: list[str],
        parameterPoint: list[float], bDebug: bool = False, bPlot: bool = False, bUseBoltzmannSuppression: bool =
        True) -> None:
    # Create the output folder if it doesn't exist already.
    pathlib.Path(str(pathlib.Path(outputFolder))).mkdir(parents=True, exist_ok=True)

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

    #potential = potentialClass(*parameterPoint[:5])
    #parameterPoint = potential.getParameterPoint()
    #np.savetxt(outputFolder + '/parameter_point.txt', parameterPoint)

    # Call PhaseTracer to determine the phase structure of the potential. 'wsl' is the Windows Subsystem for Linux,
    # which is required because PhaseTracer does not run on Windows directly. The second element of the list is the
    # program name. The remaining elements are the input parameters for the specified program. The timeout (in seconds)
    # ensures that PhaseTracer cannot run indefinitely. stdout is routed to DEVNULL to suppress any print statements
    # from PhaseTracer. stderr is routed to STDOUT so that errors in PhaseTracer are printed here.
    command = (['wsl'] if windows else []) + [PhaseTracer_directory + f'bin/{PT_script}', outputFolder +
        '/parameter_point.txt', outputFolder] + PT_params
    subprocess.call(command, timeout=60)#, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # Load the phase structure saved by PhaseTracer.
    bFileExists, phaseStructure = phase_structure.load_data(outputFolder + '/phase_structure.dat', bExpectFile=True)

    # Validate the phase structure.
    if not bFileExists:
        print('Could not find phase structure file.')
        return
    if len(phaseStructure.transitionPaths) == 0:
        print('No valid transition path to the current phase of the Universe.')
        return

    # Load and configure a PhaseHistoryAnalyser object.
    analyser = PhaseHistoryAnalyser()
    analyser.bDebug = bDebug
    analyser.bPlot = bPlot
    analyser.bReportAnalysis = bDebug
    analyser.bReportPaths = bDebug
    analyser.timeout_phaseHistoryAnalysis = 500

    # Create the potential using the parameter point.
    if potentialClass == SMplusCubic:
        potential = potentialClass(*parameterPoint, bUseBoltzmannSuppression=bUseBoltzmannSuppression)
    else:
        potential = potentialClass(*parameterPoint[:5])

    def notify_TransitionAnalyser_on_create(transitionAnalyser: TransitionAnalyser):
        transitionAnalyser.bComputeSubsampledThermalParams = True
        transitionAnalyser.bCheckPossibleCompletion = False
        transitionAnalyser.bUseChapmanJouguetVelocity = True

    notifyHandler.addEvent('TransitionAnalyser-on_create', notify_TransitionAnalyser_on_create)

    origin = np.array([0, 0])
    vev = potential.approxZeroTMin()[0]
    print('V0(0, 0)      :', potential.V0(origin))
    print('V0(vh, vs)    :', potential.V0(vev))
    print('V(0 , 0 , 0)  :', potential.Vtot(origin, 0))
    print('V(vh, vs, 0)  :', potential.Vtot(vev, 0))
    print('V(0 , 0 , 100):', potential.Vtot(origin, 100))
    print('V(vh, vs, 100):', potential.Vtot(vev, 100))

    # Analyse the phase history.
    paths, _, analysisMetrics = analyser.analysePhaseHistory_supplied(potential, phaseStructure, vw=1.)

    # Write the phase history report. Again, this will be handled within PhaseHistoryAnalysis in a future version of the
    # code.
    writePhaseHistoryReport(outputFolder + '/phase_history.json', paths, phaseStructure, analysisMetrics)


if __name__ == "__main__":
    import sys

    print(sys.argv)

    # Check that the user has included enough parameters in the run command.
    if len(sys.argv) < 4:
        print('Please specify a model, an output folder, and a parameter point.')
        sys.exit(1)

    modelLabel = sys.argv[1].lower()

    # Support model labels.
    modelLabels = ['toy', 'rss', 'rss_ht', 'archil']
    # The AnalysablePotential subclass corresponding to a particular model label.
    models = [ToyModel, RealScalarSingletModel_Boltz, RealScalarSingletModel_HT, SMplusCubic]
    # PhaseTracer script to run, specific to a particular model label.
    PT_scripts = ['run_ToyModel', 'run_RSS', 'run_RSS', 'run_supercool']
    # Extra arguments to pass to PhaseTracer, specific to a particular model label.
    PT_paramArrays = [[], ['-boltz'], ['-ht'], ['-boltz']]
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

    main(_potentialClass, outputFolder, _PT_script, _PT_params, _parameterPoint, bDebug=False, bPlot=False)
