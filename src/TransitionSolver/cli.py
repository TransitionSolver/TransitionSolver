"""
Run TransitionSolver on a known model
=====================================
"""

import logging

import click
import numpy as np
import rich
from rich.status import Status

from . import phasetracer
from .models.real_scalar_singlet_model import RealScalarSingletModel
from . import gws


np.set_printoptions(legacy='1.25')
logging.captureWarnings(True)

MODELS = {"RSS": ("run_RSS", RealScalarSingletModel)}
DETECTORS = {"LISA": gws.lisa, "LISA_SNR_10": gws.lisa_thrane_2019_snr_10}
PTAS = {"NANOGrav": gws.nanograv_15, "PPTA": gws.ppta_dr3, "EPTA": gws.epta_dr2_full}
LEVELS = {k.lower(): getattr(logging, k) for k in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]}

@click.command()
@click.option('--model', help='Model name', required=True, type=click.Choice(MODELS.keys()))
@click.option('--point', help='Parameter point file', type=click.Path(exists=True), required=True)
@click.option('--vw', default=None, help='Bubble wall velocity', type=click.FloatRange(0.))
@click.option('--detector', default=[], help='Gravitational wave detector', type=click.Choice(DETECTORS.keys()), multiple=True)
@click.option('--pta', default=[], help='Pulsar Timing Array', type=click.Choice(PTAS.keys()), multiple=True)
@click.option('--show', default=True, help='Whether to show plots', type=bool)
@click.option('--level', default="critical", help='Logging level', type=click.Choice(LEVELS.keys()))
def cli(model, point, vw, detector, pta, show, level):
    """
    Run TransitionSolver on a particular model and point
    """

    logging.getLogger().setLevel(LEVELS[level])

    program, model = MODELS[model]
    point = np.loadtxt(point)
    potential = model(*point)
   
    with Status("Running PhaseTracer"):
        phase_structure_file = phasetracer.pt_run(program, point)

    with Status("Analyzing phase history"):
        phase_history = phasetracer.phase_structure(
            potential, phase_structure_file)

    rich.print(phase_history)

    detectors = [DETECTORS[d] for d in detector]
    ptas = [PTAS[p] for p in pta]

    with Status("Analyzing gravitational wave signal"):
        analyser = gws.GWAnalyser(potential, phase_structure_file, phase_history, vw=vw)
        report = analyser.report(*detectors)

    rich.print(report)
    analyser.plot(detectors=detectors, ptas=ptas, show=show)
