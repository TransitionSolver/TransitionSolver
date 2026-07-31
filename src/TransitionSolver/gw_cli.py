"""Post-process saved transition results into gravitational-wave predictions."""

import json
import logging
from pathlib import Path

import click
import numpy as np
from . import load_potential, read_phase_tracer
from .cli import (
    DETECTORS,
    LEVELS,
    PTAS,
    analyse_and_save_gws,
    transitions_with_percolation_temperature,
    valid_transition_ids,
)
from .phasetracer import DEFAULT_NAMESPACE


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

    analyse_and_save_gws(
        potential,
        transition_report,
        phase_structure,
        transition_ids,
        valid_ids,
        detector,
        pta,
        output_folder,
        show=show,
        temperature_scan=temperature_scan,
        temperature_uncertainty=temperature_uncertainty,
        include_all_transitions_with_perc_temp=(
            include_all_transitions_with_perc_temp
        ),
    )


if __name__ == "__main__":
    gw_cli()
