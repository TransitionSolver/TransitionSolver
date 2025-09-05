"""
Run PhaseTracer 
===============
"""

import os
import subprocess
from pathlib import Path

import numpy as np

from .analysis.phase_structure import load_data
from .analysis.phase_history_analysis import PhaseHistoryAnalyser


PT_HOME = Path(os.getenv("PHASETRACER", Path.home() / ".PhaseTracer"))
PT_LIB = PT_HOME / "lib" / "libphasetracer.so"
PT_BIN = PT_HOME / "bin"
PT_UNIT_TEST = PT_BIN / "unit_tests"
WINDOWS = os.name == 'nt'
PT_DATA = Path(os.path.dirname(os.path.abspath(__file__))) / "data"


def pt_run(program, point, output_dir="output", **kwargs):
    """
    Run a PT program

    @param name Name of executable
    @param output_dir Directory for output
    @param point Parameter point

    @returns Phase structure file from PT
    """
    program = PT_BIN / program
    assert program.is_file()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    point_file = output_dir / "parameter_point.txt"
    np.savetxt(point_file, np.matrix(point))

    command = [program, point_file, output_dir]

    if WINDOWS:
        command.insert(0, "wsl")

    subprocess.check_call(command, **kwargs)

    phase_structure_file = output_dir / 'phase_structure.dat'
    assert phase_structure_file.is_file()
    return phase_structure_file


def make_report(paths, phase_structure, analysis_metrics):
    """
    @returns Report phase history from TransitionSolver objects
    """
    report = {}
    report['transitions'] = [t.getReport(None)
                             for t in phase_structure.transitions]
    report['paths'] = [p.getReport() for p in paths]
    report['valid'] = any([p.bValid for p in paths])
    report['analysisTime'] = analysis_metrics.analysisElapsedTime
    return report


def phase_structure(potential, phase_structure_file, vw=0.9):
    """
    @param potential Effective potential
    @param phase_structure_file Phase strucure results file from PT
    @param vw Assumed wall velocity

    @returns Report phase history from PhaseTracer output
    """
    phase_structure = load_data(phase_structure_file)[1]

    if not phase_structure.transitionPaths:
        raise RuntimeError(
            'No valid transition path to the current phase of the Universe')

    analyser = PhaseHistoryAnalyser()
    paths, _, analysis_metrics = analyser.analysePhaseHistory_supplied(
        potential, phase_structure)

    return make_report(paths, phase_structure, analysis_metrics)
