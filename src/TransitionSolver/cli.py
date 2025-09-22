"""
Run TransitionSolver on a known model
=====================================
"""

import click
import numpy as np
import rich

from . import phasetracer
from .models.real_scalar_singlet_model import RealScalarSingletModel
from .gws import lisa, GWAnalyser


MODELS = {"RSS": ("run_RSS", RealScalarSingletModel)}
DETECTORS = {"LISA": lisa}


@click.command()
@click.option('--model', help='Model name', required=True, type=click.Choice(MODELS.keys()))
@click.option('--point', help='Parameter point file', type=click.Path(exists=True), required=True)
@click.option('--vw', default=0.9, help='Bubble wall velocity', type=click.FloatRange(0.))
@click.option('--detector', default="LISA", help='Gravitational wave detector', type=click.Choice(DETECTORS.keys()))
@click.option('--show', default=True, help='Whether to show plots', type=bool)
def cli(model, point, vw, detector, show):
    """
    Run TransitionSolver on a particular model and point
    """
    program, model = MODELS[model]
    point = np.loadtxt(point)
    phase_structure_file = phasetracer.pt_run(program, point)
    potential = model(*point)

    # consider phase history

    phase_history = phasetracer.phase_structure(
        potential, phase_structure_file, vw=vw)

    rich.print(phase_history)

    # now consider GW spectrum

    detector = DETECTORS[detector]
    analyser = GWAnalyser(potential, phase_structure_file, phase_history)
    report = analyser.report(detector=detector)
    analyser.plot(detector=detector, show=show)
    rich.print(report)
