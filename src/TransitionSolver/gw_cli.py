"""Post-process saved transition results into gravitational-wave predictions."""

import json
import logging
from pathlib import Path

import click
import numpy as np
import rich.pretty
from rich.console import Console
from rich.status import Status
from rich.text import Text

from . import gws, load_potential, read_phase_tracer, save_gw_outputs
from .cli import (
    DETECTORS,
    LEVELS,
    PTAS,
    transition_diagnostics,
    transitions_with_percolation_temperature,
    valid_transition_ids,
)
from .phasetracer import DEFAULT_NAMESPACE


console = Console()


def load_saved_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def reconstruct_potential(results_folder: Path, saved_options: dict):
    """Reconstruct the potential used for the saved transition analysis."""
    point_file = results_folder / "parameter_point.txt"
    if not point_file.exists():
        original_point = saved_options.get("point_file_name")
        if original_point is None or not Path(original_point).exists():
            raise click.ClickException(
                "No saved parameter_point.txt was found and the original "
                "parameter-point file is unavailable."
            )
        point_file = Path(original_point)

    model = saved_options["model"]
    model_header = saved_options.get("model_header") or f"{model}.hpp"
    model_lib = saved_options.get("model_lib")
    model_namespace = saved_options.get("model_namespace") or DEFAULT_NAMESPACE
    point = np.loadtxt(point_file)
    potential_type = load_potential(
        model_header, model, model_lib, model_namespace
    )
    return potential_type(point)


@click.command()
@click.argument(
    "results_folder",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--folder",
    "output_folder",
    help="Output folder; defaults to the transition-results folder",
    type=click.Path(file_okay=False, path_type=Path),
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
@click.option("--show/--no-show", default=True, help="Whether to show plots")
@click.option(
    "--level",
    default="critical",
    help="Logging level",
    type=click.Choice(LEVELS.keys()),
)
@click.option(
    "--temperature-scan",
    help="Scan GW predictions over all valid saved temperatures",
    is_flag=True,
)
@click.option(
    "--temperature-uncertainty",
    help="Save sampled GW prediction ranges near percolation",
    is_flag=True,
)
@click.option(
    "--include-all-transitions-with-perc-temp",
    help="Also calculate GWs for every transition with a percolation temperature",
    is_flag=True,
)
def gw_cli(
    results_folder,
    output_folder,
    detector,
    pta,
    show,
    level,
    temperature_scan,
    temperature_uncertainty,
    include_all_transitions_with_perc_temp,
):
    """Calculate gravitational waves from a saved transition analysis."""
    logging.getLogger().setLevel(LEVELS[level])
    output_folder = output_folder or results_folder

    required_files = ["cli.json", "tr.json", "phasetracer.txt"]
    missing = [name for name in required_files if not (results_folder / name).exists()]
    if missing:
        raise click.ClickException(
            "Missing required transition result file(s): " + ", ".join(missing)
        )

    saved_options = load_saved_json(results_folder / "cli.json")
    transition_report = load_saved_json(results_folder / "tr.json")
    phase_structure = read_phase_tracer(
        phase_tracer_file=results_folder / "phasetracer.txt"
    )
    potential = reconstruct_potential(results_folder, saved_options)

    valid_ids = valid_transition_ids(transition_report)
    transition_ids = valid_ids
    if include_all_transitions_with_perc_temp:
        transition_ids = transitions_with_percolation_temperature(transition_report)

    if not transition_ids:
        message = "No transitions from valid cosmological paths were found."
        if not include_all_transitions_with_perc_temp:
            message += " Try --include-all-transitions-with-perc-temp."
        raise click.ClickException(message)

    diagnostics = {}
    if include_all_transitions_with_perc_temp:
        diagnostics = {
            transition_id: transition_diagnostics(
                transition_report["transitions"][transition_id]
            )
            for transition_id in transition_ids
        }
        for transition_id, diagnostic in diagnostics.items():
            for warning in diagnostic["Warnings"]:
                console.print(f"Warning for transition {transition_id}: {warning}")
            if (
                temperature_uncertainty
                and transition_report["transitions"][transition_id].get("T_f") is None
            ):
                console.print(
                    f"Warning for transition {transition_id}: The temperature-"
                    "uncertainty calculation is skipped because there is no "
                    "completion "
                    "temperature."
                )

    detectors = [DETECTORS[name] for name in detector]
    ptas = [PTAS[name] for name in pta]

    with Status("Analyzing gravitational wave signal"):
        analyser = gws.GWAnalyser(
            potential,
            transition_report,
            phase_structure,
            transition_ids=transition_ids,
        )
        gw_report = {
            transition_id: analyser.gws[transition_id].report(*detectors)
            for transition_id in transition_ids
        }
        for transition_id in diagnostics:
            gw_report[transition_id]["Transition diagnostics"] = diagnostics[
                transition_id
            ]
        gw_fig = analyser.plot(detectors=detectors, ptas=ptas, show=show)

    console.rule("[bold red]Gravitational waves")
    rich.pretty.pprint(gw_report, console=console, max_length=10)

    additional_ids = [
        transition_id
        for transition_id in transition_ids
        if transition_id not in valid_ids
    ]
    with Status("Calculating and saving gravitational wave results"):
        folder, path_dirs = save_gw_outputs(
            transition_report,
            gw_fig,
            analyser,
            detectors,
            output_folder,
            temperature_scan=temperature_scan,
            temperature_uncertainty=temperature_uncertainty,
            ptas=ptas,
            additional_transition_ids=additional_ids,
            transition_diagnostics=diagnostics,
        )

    console.print(
        Text.assemble("Gravitational wave results saved in: ", (folder, "bold magenta"))
    )
    for path in path_dirs:
        console.print(
            f"  Valid cosmological history path {path['index']}: "
            f"{path['directory']}"
        )
    for transition_id in additional_ids:
        directory = (
            Path(folder)
            / "transitions_with_perc_temp"
            / f"transition_{transition_id}"
        )
        console.print(f"  Additional transition {transition_id}: {directory}")


if __name__ == "__main__":
    gw_cli()
