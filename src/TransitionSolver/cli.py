"""
Run TransitionSolver on a known model
=====================================
"""

import logging

import ast
import click
import numpy as np
import rich
from rich.status import Status

from . import phasetracer
from . import gws
from .effective_potential import load_potential


np.set_printoptions(legacy='1.25')
logging.captureWarnings(True)

MODELS = {"RSS": "run_RSS"}
HEADERS = {"RSS": "RSS.hpp"}
DETECTORS = {"LISA": gws.lisa, "LISA_SNR_10": gws.lisa_thrane_2019_snr_10}
PTAS = {"NANOGrav": gws.nanograv_15, "PPTA": gws.ppta_dr3, "EPTA": gws.epta_dr2_full}
LEVELS = {k.lower(): getattr(logging, k) for k in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]}

@click.command()
@click.option('--model', help='Model name', required=True, type=click.Choice(MODELS.keys()))
@click.option('--model-header', help='Model header-file', required=False, default=None, type=click.Path(exists=True))
@click.option('--model-lib', help='Library for model if not header-only', required=False, default=None, type=click.Path(exists=True))
@click.option('--point', help='Parameter point file', type=click.Path(exists=True), required=True)
@click.option('--vw', default=None, help='Bubble wall velocity', type=click.FloatRange(0.))
@click.option('--detector', default=[], help='Gravitational wave detector', type=click.Choice(DETECTORS.keys()), multiple=True)
@click.option('--pta', default=[], help='Pulsar Timing Array', type=click.Choice(PTAS.keys()), multiple=True)
@click.option('--show', default=True, help='Whether to show plots', type=bool)
@click.option('--level', default="critical", help='Logging level', type=click.Choice(LEVELS.keys()))
@click.option("--apply", required=False, help="Apply settings to a potential", type=(str, ast.literal_eval), multiple=True)
def cli(model, model_header, model_lib, point, vw, detector, pta, show, level, apply):
    """
    Run TransitionSolver on a particular model and point

    Example usage:

    ts --model RSS --point rss_bp1.txt --apply set_daisy_method 2 --apply set_bUseBoltzmannSuppression True
    """

    logging.getLogger().setLevel(LEVELS[level])
    
    if model_header is None:
        model_header = HEADERS[model]

    point = np.loadtxt(point)
    
    # TODO this is very RSS specific!
    python_order = ["c1", "b3", "theta", "vs", "ms", "muh", "mus", "lh", "ls", "c2", "muh0", "mus0"]
    cpp_order = ["muh", "mus", "lh", "ls", "c1", "c2", "b3", "muh0", "mus0", "vs", "ms", "theta"]
    pt_ordered_point = [point[python_order.index(e)] for e in cpp_order]
    
    potential = load_potential(pt_ordered_point, model_header, class_name=model, lib_name=model_lib)

    for s, t in apply:
        getattr(potential, s)(t)

    program = MODELS[model]

    with Status("Running PhaseTracer"):
        phase_structure_file = phasetracer.pt_run(program, point)

    with Status("Analyzing phase history"):
        phase_history = phasetracer.phase_structure(potential, phase_structure_file)

    rich.print(phase_history)

    detectors = [DETECTORS[d] for d in detector]
    ptas = [PTAS[p] for p in pta]

    with Status("Analyzing gravitational wave signal"):
        analyser = gws.GWAnalyser(potential, phase_structure_file, phase_history, vw=vw)
        report = analyser.report(*detectors)

    rich.print(report)
    analyser.plot(detectors=detectors, ptas=ptas, show=show)
