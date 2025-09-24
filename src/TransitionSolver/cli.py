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
from .gws import nanograv_15, lisa, GWAnalyser


np.set_printoptions(legacy='1.25')
logging.captureWarnings(True)

MODELS = {"RSS": ("run_RSS", RealScalarSingletModel)}
DETECTORS = {"LISA": lisa, "none": None}
PTAS = {"NANOGrav": nanograv_15, "none": None}
LEVELS = {k.lower(): getattr(logging, k) for k in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]}

@click.command()
@click.option('--model', help='Model name', required=True, type=click.Choice(MODELS.keys()))
@click.option('--point', help='Parameter point file', type=click.Path(exists=True), required=True)
@click.option('--vw', default=0.9, help='Bubble wall velocity', type=click.FloatRange(0.))
@click.option('--detector', default="LISA", help='Gravitational wave detector', type=click.Choice(DETECTORS.keys()))
@click.option('--pta', default="NANOGrav", help='Pulsar Timing Array', type=click.Choice(PTAS.keys()))
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
            potential, phase_structure_file, vw=vw)

    rich.print(phase_history)

    with Status("Analyzing gravitational wave signal"):
        detector = DETECTORS[detector]
        pta = PTAS[pta]
        analyser = GWAnalyser(potential, phase_structure_file, phase_history)
        report = analyser.report(detector=detector)

    rich.print(report)
    analyser.plot(detector=detector, pta=pta, show=show)
