"""
Run TransitionSolver on a known model
=====================================
"""

import logging
import json
import tempfile

from pathlib import Path

import click
import numpy as np
import rich
import rich.pretty

from rich.text import Text
from rich.status import Status
from rich.console import Console

from . import (
    gws,
    load_potential,
    build_phase_tracer,
    read_phase_tracer,
    run_phase_tracer,
    find_phase_history,
    plot_summary,
    save_transition_outputs,
    save_gw_outputs
)

from .phasetracer import DEFAULT_NAMESPACE


console = Console()

np.set_printoptions(legacy="1.25")
logging.captureWarnings(True)

DETECTORS = {"LISA": gws.lisa, "LISA_SNR_10": gws.lisa_thrane_2019_snr_10}
PTAS = {"NANOGrav": gws.nanograv_15, "PPTA": gws.ppta_dr3, "EPTA": gws.epta_dr2_full}
LEVELS = {
    k.lower(): getattr(logging, k)
    for k in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
}


def deep_merge(a: dict, b: dict) -> dict:
    """Return a merged dict where b overwrites a recursively."""
    out = dict(a)
    for k, v in b.items():
        # check both values are dictionaries and recurse if true
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_json(path: str | Path) -> dict:
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_point_settings(point_settings_file, point_file_name: str) -> dict:
    """
    If no file supplied try point file name but with pt.json suffix
    """
    if point_settings_file is None:
        point_settings_file = Path(point_file_name).with_suffix(".pt.json")
    return load_json(point_settings_file)


def load_model_settings(model_settings_file, model: str) -> dict:
    """
    If no file supplied, try

    src/TransitionSolver/settings/PT_settings_<MODEL>.json
    """
    if model_settings_file is None:
        base_dir = Path(__file__).resolve().parent
        model_settings_file = base_dir / "settings" / f"PT_settings_{model}.json"
    return load_json(model_settings_file)


def create_pt_settings(
    model_settings_file, point_settings_file, model, point_file_name, *other_files
):
    """
    Create settings for PhaseTracer by resolving settings files in precedence order:

    model settings -> point settings -> repeated --pt-settings (applied last)
    """
    model_settings = load_model_settings(model_settings_file, model)
    point_settings = load_point_settings(point_settings_file, point_file_name)
    other_settings = [load_json(name) for name in other_files]

    pt_settings = model_settings.copy()
    for settings in [point_settings, *other_settings]:
        pt_settings = deep_merge(pt_settings, settings)
    return pt_settings


@click.command()
@click.option("--model", help="Model name", required=True, type=str)
@click.option(
    "--model-header",
    help="Model header-file",
    required=False,
    default=None,
    type=click.Path(exists=True),
)
@click.option(
    "--model-lib",
    help="Library for model if not header-only",
    required=False,
    default=None,
    type=click.Path(exists=True),
)
@click.option(
    "--model-namespace",
    help="Namespace for model",
    required=False,
    default=DEFAULT_NAMESPACE,
    multiple=True,
)
@click.option(
    "--point",
    "point_file_name",
    help="Parameter point file",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--vw", default=None, help="Bubble wall velocity", type=click.FloatRange(0.0)
)
@click.option(
    "--detector",
    default=[],
    help="Gravitational wave detector",
    type=click.Choice(DETECTORS.keys()),
    multiple=True,
)
@click.option(
    "--pta",
    default=[],
    help="Pulsar Timing Array",
    type=click.Choice(PTAS.keys()),
    multiple=True,
)
@click.option("--show/--no-show", default=True, help="Whether to show plots", type=bool)
@click.option(
    "--level",
    default="critical",
    help="Logging level",
    type=click.Choice(LEVELS.keys()),
)
@click.option(
    "--force",
    help="Force recompilation",
    required=False,
    default=False,
    is_flag=True,
    type=bool,
)
@click.option(
    "--action-ct",
    help="Use CosmoTransitions for action",
    required=False,
    default=False,
    is_flag=True,
    type=bool,
)
@click.option(
    "--pt-model-settings",
    help="JSON file of PhaseTracer settings for this model",
    required=False,
    default=None,
    type=click.Path(exists=True),
)
@click.option(
    "--pt-point-settings",
    help="JSON file of PhaseTracer settings for this point (overrides model settings)",
    required=False,
    default=None,
    type=click.Path(exists=True),
)
@click.option(
    "--pt-settings",
    help="Extra JSON PhaseTracer settings overrides (applied last). Can be repeated.",
    required=False,
    default=(),
    multiple=True,
    type=click.Path(exists=True),
)
@click.option(
    "--folder",
    help="Custom name of output folder",
    required=False,
    default=None,
    type=str,
)
@click.pass_context
def cli(
    ctx,
    model,
    model_header,
    model_lib,
    model_namespace,
    point_file_name,
    vw,
    detector,
    pta,
    show,
    level,
    force,
    action_ct,
    pt_model_settings,
    pt_point_settings,
    pt_settings,
    folder,
):
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
        exe_name = build_phase_tracer(
            model_header, model, model_lib, model_namespace, force
        )

    pt_settings = create_pt_settings(
        pt_model_settings, pt_point_settings, model, point_file_name, *pt_settings
    )

    with tempfile.NamedTemporaryFile(mode="w") as pt_settings_file:
        json.dump(pt_settings, pt_settings_file)
        pt_settings_file.flush()

        with Status(f"Running PhaseTracer {exe_name}"):
            phase_structure_raw = run_phase_tracer(
                exe_name, point_file_name, pt_settings_file=pt_settings_file.name
            )

    phase_structure = read_phase_tracer(phase_structure_raw)

    with Status("Analyzing phase history"):
        tr_report = find_phase_history(
            potential, phase_structure, bubble_wall_velocity=vw, action_ct=action_ct
        )
        tr_fig = plot_summary(tr_report, show=show)

    console.rule("[bold red]Transitions")
    rich.pretty.pprint(tr_report, console=console, max_length=10)

    with Status("Saving transition results"):
        folder = save_transition_outputs(
            tr_report, tr_fig, phase_structure_raw, ctx, folder
        )

    console.rule("[bold red]Results")
    console.print(
        Text.assemble("Transition results saved to: ", (folder, "bold magenta"))
    )

    if not any(path["valid"] and path["transitions"] for path in tr_report["paths"]):
        console.print(
            "No relevant transition detected in the phase history; "
            "skipping gravitational wave analysis."
        )
        return

    detectors = [DETECTORS[d] for d in detector]
    ptas = [PTAS[p] for p in pta]

    with Status("Analyzing gravitational wave signal"):
        analyser = gws.GWAnalyser(potential, tr_report, phase_structure)
        gw_report = analyser.report(*detectors)
        gw_fig = analyser.plot(detectors=detectors, ptas=ptas, show=show)

    console.rule("[bold red]Gravitational waves")
    console.print(gw_report)

    with Status("Saving gravitational wave results"):
        folder, gw_path_dirs = save_gw_outputs(
            tr_report, gw_fig, analyser, detectors, folder
        )

    console.print(
        Text.assemble(
            "Gravitational wave results also saved in: ",
            (folder, "bold magenta"),
        )
    )
    for path in gw_path_dirs:
        phases = " → ".join(str(p) for p in path["phases"])
        transitions = ", ".join(path["transitions"])
        console.print(
            Text.assemble(
                f"  Valid cosmological history path {path['index']} "
                f"(phase sequence {phases}; transition IDs {transitions}): ",
                (path["directory"], "bold magenta"),
            )
        )
