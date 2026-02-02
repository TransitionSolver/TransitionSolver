"""
Run TransitionSolver on a known model
=====================================
"""

import logging

import click
import numpy as np
import rich
import json
import os
import tempfile

from pathlib import Path
from rich.text import Text
from rich.status import Status
from rich.console import Console

from . import gws
from . import load_potential
from . import build_phase_tracer, read_phase_tracer, run_phase_tracer, find_phase_history
from . import plot_summary
from . import saveall
from .phasetracer import DEFAULT_NAMESPACE


console = Console()

np.set_printoptions(legacy='1.25')
logging.captureWarnings(True)

DETECTORS = {"LISA": gws.lisa, "LISA_SNR_10": gws.lisa_thrane_2019_snr_10}
PTAS = {"NANOGrav": gws.nanograv_15, "PPTA": gws.ppta_dr3, "EPTA": gws.epta_dr2_full}
LEVELS = {k.lower(): getattr(logging, k) for k in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]}


def deep_merge(a: dict, b: dict) -> dict:
    """Return a merged dict where b overwrites a recursively."""
    out = dict(a)
    for k, v in b.items():
        # check both values are dictionaries
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def autodetect_point_settings(point_file_name: str) -> str | None:
    p = Path(point_file_name)
    candidates = [
        str(p.with_suffix(".pt.json")),        # input/X.txt -> input/X.pt.json
        str(Path(str(p) + ".pt.json")),        # input/X.txt -> input/X.txt.pt.json
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def autodetect_model_settings(model: str) -> str | None:
    # Convention: ./pt_setings/<MODEL>.json (relative to current working dir)
    BASE_DIR = Path(__file__).resolve().parent
    c = BASE_DIR / "pt_settings" / f"{model}.json"
    if c.exists():
        print("finding specfic PT model specvifc settings for this." )
        return str(c)
    return None



@click.command()
@click.option('--model', help='Model name', required=True, type=str)
@click.option('--model-header', help='Model header-file', required=False, default=None, type=click.Path(exists=True))
@click.option('--model-lib', help='Library for model if not header-only', required=False, default=None, type=click.Path(exists=True))
@click.option('--model-namespace', help='Namespace for model', required=False, default=DEFAULT_NAMESPACE, multiple=True)
@click.option('--point', 'point_file_name', help='Parameter point file', type=click.Path(exists=True), required=True)
@click.option('--vw', default=None, help='Bubble wall velocity', type=click.FloatRange(0.))
@click.option('--detector', default=[], help='Gravitational wave detector', type=click.Choice(DETECTORS.keys()), multiple=True)
@click.option('--pta', default=[], help='Pulsar Timing Array', type=click.Choice(PTAS.keys()), multiple=True)
@click.option('--show/--no-show', default=True, help='Whether to show plots', type=bool)
@click.option('--level', default="critical", help='Logging level', type=click.Choice(LEVELS.keys()))
@click.option('--force', help='Force recompilation', required=False, default=False, is_flag=True, type=bool)
@click.option('--action-ct', help='Use CosmoTransitions for action', required=False, default=False, is_flag=True, type=bool)
@click.option('--pt-model-settings', help='JSON file of PhaseTracer settings for this model',
              required=False, default=None, type=click.Path(exists=True))
@click.option('--pt-point-settings', help='JSON file of PhaseTracer settings for this point (overrides model settings)',
              required=False, default=None, type=click.Path(exists=True))
@click.option('--pt-settings', help='Extra JSON PhaseTracer settings overrides (applied last). Can be repeated.',
              required=False, default=(), multiple=True, type=click.Path(exists=True))
@click.pass_context
def cli(ctx, model, model_header, model_lib, model_namespace, point_file_name, vw, detector, pta, show, level, force, action_ct, pt_model_settings, pt_point_settings, pt_settings):
    """
    Run TransitionSolver on a particular model and point

    Example usage:

    ts --model RSS_BP --point input/RSS/RSS_BP1.txt
    """
    logging.getLogger().setLevel(LEVELS[level])

    if model_header is None:
        model_header = f"{model}.hpp"

    point = np.loadtxt(point_file_name)
    potential = load_potential(model_header, model, model_lib, model_namespace)(point)

    with Status(f"Building PhaseTracer {model}"):
        exe_name = build_phase_tracer(model_header, model, model_lib, model_namespace, force)

    # Resolve settings files in precedence order:
    # model settings -> point settings -> repeated --pt-settings (applied last)
    model_cfg = pt_model_settings or autodetect_model_settings(model)
    point_cfg = pt_point_settings or autodetect_point_settings(point_file_name)

    effective_cfg: dict = {}
    for fp in [model_cfg, point_cfg, *pt_settings]:
        if fp:
            effective_cfg = deep_merge(effective_cfg, load_json(fp))

    pt_settings_tmp = None
    pt_settings_file = None

    try:
        if effective_cfg:
            pt_settings_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            json.dump(effective_cfg, pt_settings_tmp, indent=2, sort_keys=True)
            pt_settings_tmp.flush()
            pt_settings_tmp.close()
            pt_settings_file = pt_settings_tmp.name

        
        with Status(f"Running PhaseTracer {exe_name}"):
            phase_structure_raw = run_phase_tracer(
                exe_name,
                point_file_name,
                pt_settings_file=pt_settings_file
            )
    finally: 
        if pt_settings_file is not None:
            try:
                os.unlink(pt_settings_file)
            except OSError:
                pass
    phase_structure = read_phase_tracer(phase_structure_raw)

    with Status("Analyzing phase history"):
        tr_report = find_phase_history(potential, phase_structure, bubble_wall_velocity=vw, action_ct=action_ct)
        tr_fig = plot_summary(tr_report, show=show)

    console.rule("[bold red]Transitions")
    rich.pretty.pprint(tr_report, console=console, max_length=10)

    detectors = [DETECTORS[d] for d in detector]
    ptas = [PTAS[p] for p in pta]

    with Status("Analyzing gravitational wave signal"):
        analyser = gws.GWAnalyser(potential, phase_structure, tr_report, is_file=False)  # TODO remove is_file
        gw_report = analyser.report(*detectors)
        gw_fig = analyser.plot(detectors=detectors, ptas=ptas, show=show)

    console.rule("[bold red]Gravitational waves")
    console.print(gw_report)

    with Status("Saving results"):
        folder = saveall(tr_report, gw_report, tr_fig, gw_fig, ctx)

    console.rule("[bold red]Results")
    console.print(Text.assemble("Results saved to: ", (folder, "bold magenta")))
