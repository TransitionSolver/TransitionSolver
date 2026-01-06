"""
Run TransitionSolver on a known model
=====================================
"""

import logging

import ast
import click
import numpy as np
import rich
from rich.text import Text
from rich.status import Status

from . import gws
from . import load_potential
from . import build_phase_tracer, read_phase_tracer, run_phase_tracer, find_phase_history
from . import plot_summary
from . import saveall


np.set_printoptions(legacy='1.25')
logging.captureWarnings(True)

DETECTORS = {"LISA": gws.lisa, "LISA_SNR_10": gws.lisa_thrane_2019_snr_10}
PTAS = {"NANOGrav": gws.nanograv_15, "PPTA": gws.ppta_dr3, "EPTA": gws.epta_dr2_full}
LEVELS = {k.lower(): getattr(logging, k) for k in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]}


@click.command()
@click.option('--model', help='Model name', required=True, type=str)
@click.option('--model-header', help='Model header-file', required=False, default=None, type=click.Path(exists=True))
@click.option('--model-lib', help='Library for model if not header-only', required=False, default=None, type=click.Path(exists=True))
@click.option('--model-namespace', help='Namespace for model', required=False, default="EffectivePotential", type=str)
@click.option('--point', 'point_file_name', help='Parameter point file', type=click.Path(exists=True), required=True)
@click.option('--vw', default=None, help='Bubble wall velocity', type=click.FloatRange(0.))
@click.option('--detector', default=[], help='Gravitational wave detector', type=click.Choice(DETECTORS.keys()), multiple=True)
@click.option('--pta', default=[], help='Pulsar Timing Array', type=click.Choice(PTAS.keys()), multiple=True)
@click.option('--show/--no-show', default=True, help='Whether to show plots', type=bool)
@click.option('--level', default="critical", help='Logging level', type=click.Choice(LEVELS.keys()))
@click.option("--apply", required=False, help="Apply settings to a potential", type=(str, ast.literal_eval), multiple=True)
@click.option('--force', help='Force recompilation', required=False, default=False, is_flag=True, type=bool)
@click.option('--action-ct', help='Use CosmoTransitions for action', required=False, default=False, is_flag=True, type=bool)
@click.option('--t-high', help='High temperature to consider in PhaseTracer', required=False, default=1e3, type=float)
@click.pass_context
def cli(ctx, model, model_header, model_lib, model_namespace, point_file_name, vw, detector, pta, show, level, apply, force, action_ct, t_high):
    """
    Run TransitionSolver on a particular model and point

    Example usage:

    ts --model RSS --point input/RSS/RSS_BP1.txt --apply set_daisy_method 2 --apply set_bUseBoltzmannSuppression True --force
    """

    logging.getLogger().setLevel(LEVELS[level])

    if model_header is None:
        model_header = f"{model}.hpp"

    point = np.loadtxt(point_file_name)
    potential = load_potential(point, model_header, class_name=model, lib_name=model_lib)

    for s, t in apply:
        getattr(potential, s)(t)

    with Status(f"Building PhaseTracer {model}"):
        exe_name = build_phase_tracer(model, model_header, model_lib, model_namespace, force)

    with Status(f"Running PhaseTracer {exe_name}"):
        phase_structure_raw = run_phase_tracer(exe_name, point_file_name, t_high)
        phase_structure = read_phase_tracer(phase_structure_raw)

    with Status("Analyzing phase history"):
        tr_report = find_phase_history(potential, phase_structure, vw=vw, action_ct=action_ct)
        tr_fig = plot_summary(tr_report, show=show)

    rich.print(tr_report)
    
    detectors = [DETECTORS[d] for d in detector]
    ptas = [PTAS[p] for p in pta]

    with Status("Analyzing gravitational wave signal"):
        analyser = gws.GWAnalyser(potential, phase_structure, tr_report, is_file=False)  # TODO remove is_file
        gw_report = analyser.report(*detectors)
        gw_fig = analyser.plot(detectors=detectors, ptas=ptas, show=show)

    rich.print(gw_report)

    with Status("Saving results"):
        folder = saveall(tr_report, gw_report, tr_fig, gw_fig, ctx)

    rich.print(Text.assemble("Results saved to: ", (folder, "bold magenta")))
