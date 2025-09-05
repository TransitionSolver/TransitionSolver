"""
Run TransitionSolver on a known model
=====================================
"""

import click
import numpy as np
import rich

from . import phasetracer
from .models.real_scalar_singlet_model import RealScalarSingletModel


MODELS = {"RSS": ("run_RSS", RealScalarSingletModel)}

@click.command()
@click.option('--model', help='Model name', required=True, type=click.Choice(MODELS.keys()))
@click.option('--point', help='Parameter point file', type=click.Path(exists=True), required=True)
@click.option('--vw', default=0.9, help='Bubble wall velocity', type=click.FloatRange(0.))
def cli(model, point, vw):
    """
    Run TransitionSolver on a particular model and point
    """
    program, model = MODELS[model]
    point = np.loadtxt(point)
    phase_structure_file = phasetracer.pt_run(program, point)
    potential = model(*point)
    result = phasetracer.phase_structure(potential, phase_structure_file, vw=vw)
    
    rich.print(result)
