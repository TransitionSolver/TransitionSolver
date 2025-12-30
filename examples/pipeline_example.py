# This is a stripped back version of some of the scanning pipeline I used in the supercool subtleties paper. By passing
# a simple index parameter into the pipeline and using that index to determine the parameter point and file names, one
# can run multiple instances of the pipeline in parallel with different indices.

from __future__ import annotations
import numpy as np

from typing import Type, Callable, Optional

# For handling file paths.
import pathlib
# For running PhaseTracer.
import subprocess
# For deleting files.
import shutil
# For suppressing warnings from CosmoTransitions regarding numerical overflow.
import warnings
# For reading/writing JSON files (the phase history reports).
import json
# For printing exceptions.
import traceback
# For getting the current function name.
import inspect
# For exiting the program early if necessary.
import sys

from src.TransitionSolver.models.toy_model import ToyModel
from src.TransitionSolver.analysis.phase_structure import PhaseStructure
from src.TransitionSolver.analysis.phase_history_analysis import PhaseHistoryAnalyser
from src.TransitionSolver.analysis.transition_graph import Path
from src.TransitionSolver.analysis.transition_analysis import TransitionAnalyser, ActionSampler
from src.TransitionSolver.analysis import phase_structure
from src.TransitionSolver.models.analysable_potential import AnalysablePotential
from src.TransitionSolver.util.events import notifyHandler
from TransitionSolver import read_phase_tracer

class PipelineSettings:
    bDebug: bool = False
    bPlot: bool = False
    bReportAnalysis: bool = False
    bReportPaths: bool = False
    bCleanupInvalidPoints: bool = False
    bCleanupIrrelevantPoints: bool = False
    bStoreInvalidPotential: bool = False
    bPreExistingResult: bool = False
    bSkipAnalysis: bool = False

    timeout_phaseStructure: float = 10.
    timeout_phaseHistoryAnalysis: float = 200.
    fileName_precomputedActionCurve: list[str] = []
    precomputedTransitionIDs: list[int] = []
    function_getParameterPoint: Optional[Callable[..., list[float]]]
    function_isValidPotential: Optional[Callable[[AnalysablePotential], bool]]
    function_isPhaseStructureRelevant: Callable[[PhaseStructure], bool]
    function_getWallVelocity: Callable[..., float]
    fileName_parameterPoint: str = ''
    fileName_phaseStructure: str = ''
    fileName_phaseHistoryReport: str = ''
    potentialClass: Type[AnalysablePotential]
    phaseStructureProgram_name: str = ''
    phaseStructureProgram_commands: list[str] = []

    def fillPhaseHistoryAnalyserSettings(self, pha: PhaseHistoryAnalyser):
        pha.bDebug = self.bDebug
        pha.bPlot = self.bPlot
        pha.bReportAnalysis = self.bReportAnalysis
        pha.bReportPaths = self.bReportPaths
        pha.timeout_phaseHistoryAnalysis = self.timeout_phaseHistoryAnalysis
        pha.fileName_precomputedActionCurve = self.fileName_precomputedActionCurve
        pha.precomputedTransitionIDs = self.precomputedTransitionIDs

    def setFileNames(self, parameterPoint: str, phaseStructure: str, phaseHistoryReport: str):
        self.fileName_parameterPoint = parameterPoint
        self.fileName_phaseStructure = phaseStructure
        self.fileName_phaseHistoryReport = phaseHistoryReport

    def setPrecomputedTransitions(self, fileNames: list[str], IDs: list[int]):
        self.fileName_precomputedActionCurve = fileNames
        self.precomputedTransitionIDs = IDs

    def setFunctions(self, getParameterPoint: Optional[Callable[..., list[float]]], isValidPotential:
            Optional[Callable[[AnalysablePotential], bool]], isPhaseStructureRelevant:
            Callable[[PhaseStructure], bool], getWallVelocity: Callable[..., float]):
        self.function_getParameterPoint = getParameterPoint
        self.function_isValidPotential = isValidPotential
        self.function_isPhaseStructureRelevant = isPhaseStructureRelevant
        self.function_getWallVelocity = getWallVelocity

    def setPhaseStructureCall(self, programName, programCommands):
        self.phaseStructureProgram_name = programName
        self.phaseStructureProgram_commands = programCommands


def pipeline_full(settings: PipelineSettings):
    # Prerequisites:
    #   1. A potential class defined in TransitionSolver (extending AnalysablePotential).
    #   2. A potential class defined in PhaseTracer (extending Potential or OneLoopPotential).

    # Step 1: Create a potential in TransitionSolver. Write the parameter point to a file. Alternatively, this step
    #   could be done in PhaseTracer.
    # Step 2: Determine the phase structure using PhaseTracer. Read the parameter point saved to a file to initialise
    #   the potential in PhaseTracer. Run PhaseTracer by calling PhaseFinder.find_phases,
    #   TransitionFinder.find_transitions and TransitionFinder.find_transition_paths. Save the output to a file.
    # Step 3: Load the phase structure into TransitionSolver. Determine the phase history using TransitionSolver. Read
    #   the phase structure saved to a file. Run TransitionSolver by calling PhaseHistoryAnalyser.analysePhaseHistory
    #   if you have the phase structure's file name, or PhaseHistoryAnalyser.analysePhaseHistory_supplied if you have
    #   already loaded the phase structure. Save the output to a file.

    # Note: There are a few steps along the way where the pipeline may be halted. If the potential constructed from a
    # sampled parameter point is invalid (e.g. non-perturbative, or not consistent with some other user-defined
    # constraint), the phase structure and phase history are not determined. If the potential is valid but the phase
    # structure is deemed to be invalid or irrelevant (e.g. the high-temperature vacuum is always stable, or there are
    # no first-order phase transitions), the phase history is not determined. This avoids wasted effort. Similarly,
    # within the phase history analysis itself, the analysis is halted as soon as the history is found to be
    # inconsistent with our current vacuum, and the individual transition analyses are halted as soon as the transitions
    # are determined to not complete. Further, some transitions may not be analysed if they are not relevant to the
    # phase history (e.g. due to competition with another successful transition).

    # Ignore overflow warnings. They arise from CosmoTransitions.
    warnings.filterwarnings("ignore", message="overflow encountered")

    # We don't need the second return value, numAttempts.
    potential, _ = pipeline_createPotential(settings)

    if potential is None:
        return 'Invalid potential'

    return pipeline_potentialSupplied(potential, settings)


# Attempts to load the parameter point from a file if we are expecting a pre-existing result, otherwise samples a
# parameter point using settings.function_getParameterPoint. Returns the potential (or None if failed to load from a
# pre-existing result or resampling is not enabled) and the number of parameter points trialed before a valid potential
# was found.
def pipeline_createPotential(settings: PipelineSettings, bResampleIfInvalid: bool = False)\
        -> (Optional[AnalysablePotential], int):
    numSamples = 0
    fileName = settings.fileName_parameterPoint

    if settings.bPreExistingResult:
        try:
            parameterPoint = np.loadtxt(fileName)
            return settings.potentialClass(*parameterPoint), numSamples
        except OSError:
            print('Could not find or open:', fileName)
            return None, numSamples

    while True:
        numSamples += 1
        # Create the potential.
        parameterPoint = settings.function_getParameterPoint()
        potential = settings.potentialClass(*parameterPoint)

        # If the potential is invalid (e.g. non-perturbative), we may wish to resample until we find a valid potential.
        # Otherwise, return None.
        if not settings.function_isValidPotential(potential):
            if bResampleIfInvalid:
                continue
            elif settings.bStoreInvalidPotential:
                np.savetxt(fileName, np.array(potential.getParameterPoint()))

            return None, numSamples

        break

    # Create the folder where we will save the parameter point.
    # parents[0] removes the file name from the path.
    # See https://stackoverflow.com/questions/35490148/how-to-get-folder-name-in-which-given-file-resides-from-pathlib-path
    pathlib.Path(str(pathlib.Path(fileName).parents[0])).mkdir(parents=True, exist_ok=True)
    np.savetxt(fileName, np.array([potential.getParameterPoint()]))

    return potential, numSamples


# See pipeline_full for details. This is the same, except the potential is already supplied and so does not need to be
# created and validated.
def pipeline_potentialSupplied(potential: AnalysablePotential, settings: PipelineSettings):
    message, phaseStructure = pipeline_getPhaseStructure(settings)

    fileName = settings.fileName_phaseStructure

    if phaseStructure is None:
        print('phase structure is empty')
        if settings.bCleanupInvalidPoints:
            # Attempt to delete the invalid point.
            try:
                shutil.rmtree(pathlib.Path(fileName).parents[0])
            except OSError:
                traceback.print_exc()

        return message

    isRelevant = settings.function_isPhaseStructureRelevant(phaseStructure)

    if not isRelevant:
        # Attempt to delete the irrelevant point.
        if settings.bCleanupIrrelevantPoints:
            try:
                shutil.rmtree(pathlib.Path(fileName).parents[0])
            except OSError:
                traceback.print_exc()

        return 'Irrelevant phase structure'

    if not settings.bSkipAnalysis:
        print('analysing...')
        return pipeline_analysePhaseHistory(potential, phaseStructure, settings)
    else:
        print('skipping analysis')
        return 'Success'


def pipeline_getPhaseStructure(settings: PipelineSettings):
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

    try:
        if not settings.bPreExistingResult:
            # Call PhaseTracer. Suppress standard output from PhaseTracer.
            command = (['wsl'] if windows else []) + [PhaseTracer_directory + 'bin/' +
                settings.phaseStructureProgram_name, *settings.phaseStructureProgram_commands]
            subprocess.call(command, timeout=settings.timeout_phaseStructure, stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT)

        # Check if the output file exists.
        phaseStructure = read_phase_tracer(phase_tracer_file=settings.fileName_phaseStructure)   

    #     if not bFileExists:
    #         return 'Invalid phase structure', None

        return 'Success', phaseStructure
    except subprocess.TimeoutExpired:
        return 'Phase structure extraction timed out', None


def pipeline_analysePhaseHistory(potential, phaseStructure, settings: PipelineSettings):
    analyser = PhaseHistoryAnalyser()
    settings.fillPhaseHistoryAnalyserSettings(analyser)

    
    vw = settings.function_getWallVelocity()
    paths, timedOut, _ = analyser.analysePhaseHistory_supplied(potential, phaseStructure, vw=vw)
    if timedOut:
        return 'Phase history analysis timed out'

    writePhaseHistoryReport(paths, phaseStructure, settings)

    return 'Success'
    # catch errors like this to preserve scans, just get rid of try and catch if debugging to get the stack traces. 
    
    # try:
    #     vw = settings.function_getWallVelocity()
    #     print("In pipeline_analysePhaseHistory about to call analyser.analysePhaseHistory_supplied")
    #     paths, timedOut = analyser.analysePhaseHistory_supplied(potential, phaseStructure, vw=vw)
    #     print("In pipeline_analysePhaseHistory immediately after calllling analyser.analysePhaseHistory_supplied")
    #     if timedOut:
    #         return 'Phse history analysis timed out'

    #     print("In pipeline_analysePhaseHistory about to call writePhaseHistoryReport")
    #     writePhaseHistoryReport(paths, phaseStructure, settings)

    #     return 'Success'
    # # catch errors like this to preserve scans, just get rid of try and catch if debugging to get the stack traces. 
    # except Exception as e:
    #     return f'Phase history analysis failed: {e}'
    


def writePhaseHistoryReport(paths: list[Path], phaseStructure: PhaseStructure, settings: PipelineSettings):
    report = {}
    fileName = settings.fileName_phaseHistoryReport
    print("In writePhaseHistoryReport")
    if len(phaseStructure.transitions) > 0:
        report['transitions'] = [t.report() for t in phaseStructure.transitions]
    print("In writePhaseHistoryReport after checking len(phaseStructure.transitions)")    
    if len(paths) > 0:
        print("In writePhaseHistoryReport checking len(paths)...")
        print(" len(paths) = ", len(paths))
        print(" paths = ", paths)
        report['paths'] = [p.report() for p in paths]
    print("In writePhaseHistoryReport after checking len(paths)")    
    report['valid'] = any([p.is_valid for p in paths])
    #report['analysisTime'] = analysisMetrics.analysisElapsedTime

    if settings.bDebug:
        print('Writing report...')

    try:
        with open(fileName, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4, default=str)
    except (json.decoder.JSONDecodeError, TypeError):
        print('We have a JSON serialisation error. The report is:')
        print(report)
        print('Failed to write report.')


def notify_ActionSampler_on_create(actionSampler: ActionSampler):
    if type(actionSampler) != ActionSampler:
        raise Exception(f'<{inspect.currentframe().f_code.co_name}> Expected type: TransitionAnalysis.ActionSampler,'
            f' instead got type: {type(actionSampler).__name__}')

    # The step size doesn't have a huge impact on the number of action samples because it is a remnant of an old
    # sampling algorithm, and is not properly utilised in the new sampling algorithm. Nevertheless, a smaller value will
    # result in less action samples. It relates to the ratio between the previous and next action sample values, and
    # consequently governs the separation between temperature samples. It is the maximum deviation from a unit ratio, so
    # stepSizeMax=0.5 would mean that ratios above 0.5 are allowed but ratios below 0.5 are not allowed.
    # If the action sampling needs to be adjusted, it would be best to contact me to update the algorithm itself rather
    # than trying to adjust this parameter to extreme values.
    actionSampler.stepSizeMax = 0.9
    actionSampler.actionTolerance = 1e-6
    actionSampler.bForcePhaseOnAxis = False


def notify_TransitionAnalyser_on_create(transitionAnalyser: TransitionAnalyser):
    if type(transitionAnalyser) != TransitionAnalyser:
        raise Exception(f'<{inspect.currentframe().f_code.co_name}> Expected type: TransitionAnalysis.TransitionAnalyse'
                        f'r, instead got type: {type(transitionAnalyser).__name__}')

    transitionAnalyser.bCheckPossibleCompletion = True
    transitionAnalyser.bAnalyseTransitionPastCompletion = False
    transitionAnalyser.bAllowErrorsForTn = True


def notify_PhaseHistoryAnalyser_on_create(phaseHistoryAnalyser: PhaseHistoryAnalyser):
    if type(phaseHistoryAnalyser) != PhaseHistoryAnalyser:
        raise Exception(f'<{inspect.currentframe().f_code.co_name}> Expected type: PhaseHistoryAnalysis.PhaseHistoryAna'
            f'lyser, instead got type: {type(phaseHistoryAnalyser).__name__}')

    pass


def generateParameterPoint(fileName):
    parameterPoint = np.array([0.104005, 250, 3.5, 0.2])
    np.savetxt(fileName, parameterPoint)


# An example where we first construct a potential manually, then run PhaseTracer and TransitionSolver on that potential.
# The parameter values that define the potential are not saved.
# TODO: run_ToyModel no longer accepts individual parameter values so this method is now irrelevant.
"""def example():
    # Set up notification events for convenient configuration of other objects required for the analysis. E.g. when the
    # ActionSampler object is created, call notify_ActionSampler_on_create, passing in the ActionSampler instance so its
    # properties can be configured. This avoids the need for passing large sets of parameters through multiple objects.
    notifyHandler.addEvent('ActionSampler-on_create', notify_ActionSampler_on_create)
    notifyHandler.addEvent('TransitionAnalyser-on_create', notify_TransitionAnalyser_on_create)
    notifyHandler.addEvent('PhaseHistoryAnalyser-on_create', notify_PhaseHistoryAnalyser_on_create)

    # Create a potential to analyse.
    potential = ToyModel(0.104005, 250, 3.5, 0.2)

    # ==================================================================================================================
    # Define configurations for the program.
    # ==================================================================================================================

    # This is the name of the PhaseTracer program that will be executed to determine the phase structure. This program
    # name will be searched for in <path_to_PhaseTracer>/PhaseTracer/bin/.
    phaseStructureProgramName = 'run_ToyModel'
    # Where all output files will be saved, relative to the TransitionSolver directory.
    outputFolder = 'output/example1'
    # Write PhaseTracer's output to the output folder.
    phaseStructureOutputFolder = outputFolder
    # The name of the phase structure file to search for. PhaseTracer will name the file phase_structure.dat by default.
    fileName_phaseStructure = outputFolder + '/phase_structure.dat'
    # The name of the phase history report. This is user defined.
    fileName_phaseHistoryReport = outputFolder + '/phase_history.json'
    # The commands to pass to the PhaseTracer program. In this case, it is the parameter values that define the
    # potential.
    phaseStructureProgramCommands = [str(potential.AonV), str(potential.v), str(potential.D), str(potential.E),
        phaseStructureOutputFolder]
    # Whether the phase structure is relevant for the current study. E.g. we require at least one transition path from
    # the high-temperature phase to the current phase of the Universe.
    function_isPhaseStructureRelevant = lambda phaseStructure: len(phaseStructure.transitionPaths) > 0
    # Currently, TransitionSolver only handles a constant wall velocity. We may want the wall velocity to depend on some
    # scan variable (or be the scan variable itself), hence we set it through a function which could take arguments.
    function_getWallVelocity = lambda: 1.

    # ==================================================================================================================
    # Apply the configurations.
    # ==================================================================================================================

    settings = PipelineSettings()
    # We don't need to provide a file name for the parameter point since we don't save that.
    settings.setFileNames('', fileName_phaseStructure, fileName_phaseHistoryReport)
    # We don't need to provide a function to generate the parameter point or determine whether the potential is valid.
    settings.setFunctions(None, None, function_isPhaseStructureRelevant, function_getWallVelocity)
    settings.setPhaseStructureCall(phaseStructureProgramName, phaseStructureProgramCommands)
    settings.bDebug = True
    settings.bPlot = True
    settings.bReportAnalysis = True
    settings.bReportPaths = True
    settings.bCheckPossibleCompletion = False

    print('About to run.')

    # Runs PhaseTracer and TransitionSolver on the potential.
    pipeline_potentialSupplied(potential, settings)

    print('Finished')"""


# An example where we define a function that constructs a potential and saves the parameter values to a file. Then we
# run PhaseTracer and TransitionSolver on that potential. This method is more convenient for redoing a scan over the
# parameter space where the same points should be used. A significant amount of time can be saved if the previous
# action samples can be reused in the new scan.
def example_parameterPointFile():
    # Set up notification events for convenient configuration of other objects required for the analysis. E.g. when the
    # ActionSampler object is created, call notify_ActionSampler_on_create, passing in the ActionSampler instance so its
    # properties can be configured. This avoids the need for passing large sets of parameters through multiple objects.
    notifyHandler.addEvent('ActionSampler-on_create', notify_ActionSampler_on_create)
    notifyHandler.addEvent('TransitionAnalyser-on_create', notify_TransitionAnalyser_on_create)
    notifyHandler.addEvent('PhaseHistoryAnalyser-on_create', notify_PhaseHistoryAnalyser_on_create)

    # ==================================================================================================================
    # Define configurations for the program.
    # ==================================================================================================================

    # This is the name of the PhaseTracer program that will be executed to determine the phase structure. This program
    # name will be searched for in <path_to_PhaseTracer>/PhaseTracer/bin/.
    phaseStructureProgramName = 'run_ToyModel'
    # Where all output files will be saved, relative to the TransitionSolver directory.
    outputFolder = 'output/pipeline_example_output'
    # Write PhaseTracer's output to the output folder.
    phaseHistoryOutputFolder = outputFolder

    fileName_parameterPoint = outputFolder + '/parameter_point.txt'
    # The name of the phase structure file to search for. PhaseTracer will name the file phase_structure.dat by default.
    fileName_phaseStructure = outputFolder + '/phase_structure.dat'
    # The name of the phase history report. This is user defined.
    fileName_phaseHistoryReport = outputFolder + '/phase_history.json'
    # The commands to pass to the PhaseTracer program. In this case, it is the parameter values that define the
    # potential.
    phaseStructureProgramCommands = [fileName_parameterPoint, phaseHistoryOutputFolder]
    # The function that generates parameter points. This could take arguments (such as point ID) to define a sampling
    # scheme for a scan.
    function_getParameterPoint = lambda: [0.104005, 250, 3.5, 0.2]
    # The function that determines whether a potential is valid (i.e. satisfies some user-defined constraints).
    function_isValidPotential = lambda potential: potential.AonV > 0 and potential.v > 0 and potential.D > 0 and\
        potential.E > 0
    # Whether the phase structure is relevant for the current study. E.g. we require at least one transition path from
    # the high-temperature phase to the current phase of the Universe.
    function_isPhaseStructureRelevant = lambda phaseStructure: len(phaseStructure.paths) > 0
    # Currently, TransitionSolver only handles a constant wall velocity. We may want the wall velocity to depend on some
    # scan variable (or be the scan variable itself), hence we set it through a function which could take arguments.
    function_getWallVelocity = lambda: 1.

    # ==================================================================================================================
    # Apply the configurations.
    # ==================================================================================================================

    settings = PipelineSettings()
    settings.setFileNames(fileName_parameterPoint, fileName_phaseStructure, fileName_phaseHistoryReport)
    settings.setFunctions(function_getParameterPoint, function_isValidPotential, function_isPhaseStructureRelevant,
        function_getWallVelocity)
    settings.setPhaseStructureCall(phaseStructureProgramName, phaseStructureProgramCommands)
    settings.bDebug = True
    settings.bPlot = False
    settings.bReportAnalysis = True
    settings.bReportPaths = True
    settings.bCheckPossibleCompletion = False
    # This time we have to set the potential class.
    settings.potentialClass = ToyModel

    # Creates a potential, then runs PhaseTracer and TransitionSolver on the potential.
    pipeline_full(settings)


if __name__ == "__main__":
    example_parameterPointFile()
