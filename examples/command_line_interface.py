from __future__ import annotations
import traceback

from TransitionSolver.analysis.transition_analysis import TransitionAnalyser
from TransitionSolver.models.supercool_model import SMplusCubic
from TransitionSolver.models.analysable_potential import AnalysablePotential
from TransitionSolver.models.toy_model import ToyModel
from TransitionSolver.models.real_scalar_singlet_model_boltz import RealScalarSingletModel_Boltz
from TransitionSolver.models.real_scalar_singlet_model_ht import RealScalarSingletModel_HT
from TransitionSolver.analysis.phase_structure import PhaseStructure
from TransitionSolver.analysis.phase_history_analysis import AnalysisMetrics, PhaseHistoryAnalyser
from TransitionSolver.analysis.transition_graph import Path
from TransitionSolver.analysis import phase_structure
from TransitionSolver.gws import GWAnalyser, lisa
from TransitionSolver import read_phase_tracer
from typing import Type
import numpy as np
import subprocess
import json
import pathlib
import sys

from TransitionSolver.util.events import notifyHandler


def writePhaseHistoryReport(fileName: str, paths: list[Path], phaseStructure: PhaseStructure, analysisMetrics:
        AnalysisMetrics) -> None:
    report = {}

    if len(phaseStructure.transitions) > 0:
        report['transitions'] = [t.report() for t in phaseStructure.transitions]
    if len(paths) > 0:
        report['paths'] = [p.report() for p in paths]
    report['valid'] = any([p.is_valid for p in paths])
    report['analysisTime'] = analysisMetrics.analysisElapsedTime

    print('Writing report...')

    try:
        with open(fileName, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4, default=str)
    except (json.decoder.JSONDecodeError, TypeError):
        print('We have a JSON serialisation error. The report is:')
        print(report)
        print('Failed to write report.')

    return report

def main(potentialClass: Type[AnalysablePotential], GWs: int, outputFolder: str, PT_script: str, PT_params: list[str],
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
    if bDebug == True:
        print("Calling PhaseTracer with command ",  command)
    subprocess.call(command, timeout=60)#, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # Load the phase structure saved by PhaseTracer.
    phaseStructure = read_phase_tracer(phase_tracer_file=outputFolder + '/phase_structure.dat')

    if not phaseStructure.paths:
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
    # boolean value determines if we use the chapman Jouguet velocity of the vw passed
    # to analysePhaseHistory_supplied if True the vw passed to analysePhaseHistory_supplied
    # is not used
    bUseCJvw = True
    def notify_TransitionAnalyser_on_create(transitionAnalyser: TransitionAnalyser):
        transitionAnalyser.bComputeSubsampledThermalParams = True
        transitionAnalyser.bCheckPossibleCompletion = False
        transitionAnalyser.bUseChapmanJouguetVelocity = bUseCJvw

    notifyHandler.addEvent('TransitionAnalyser-on_create', notify_TransitionAnalyser_on_create)

    origin = np.array([0, 0])
    vev = potential.approxZeroTMin()[0]
    # Analyse the phase history.
    paths, _, analysisMetrics = analyser.analysePhaseHistory_supplied(potential, phaseStructure, vw=0.96)

    # Write the phase history report. Again, this will be handled within PhaseHistoryAnalysis in a future version of the
    # code.
    report = writePhaseHistoryReport(outputFolder + '/phase_history.json', paths, phaseStructure, analysisMetrics)
    
    if GWs == 0:
        return
    elif GWs == 1:
        analyser = GWAnalyser(potential, outputFolder + '/phase_structure.dat', report, kappa_turb=0.05)  
    elif GWs == 2:
        analyser = GWAnalyser(potential, outputFolder + '/phase_structure.dat', report, kappa_coll=1)
          
    report = analyser.report(lisa)
    print(report)

if __name__ == "__main__":
    import sys

    print(sys.argv)

    # Check that the user has included enough parameters in the run command.
    if len(sys.argv) < 5:
        print('Please specify a model (e.g. toy, rss, rss_ht, smpluscubic) whether you want GWs computed (0 no, 1 GWs sourced from sound waves and turbulence, see README for other options), an output folder, and a parameter point.')
        sys.exit(1)

    modelLabel = sys.argv[1].lower()

    # Support model labels.
    modelLabels = ['toy', 'rss', 'rss_ht', 'smpluscubic']
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

    # Check for GWs
    # if 0 no GWs, if 1 compute GWs for sound waves and turbulence
    # if 2 compute for collsions
    # if 3 compute for soundaves + turbulence and for collsions separately
    GWs = int(sys.argv[2])

    outputFolder = sys.argv[3]

    _parameterPoint = sys.argv[4]
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

    main(_potentialClass, GWs, outputFolder, _PT_script, _PT_params, _parameterPoint, bDebug=False, bPlot=False)
