"""
Make a report of results
========================
"""

import time
import os
from pathlib import Path

import json
import matplotlib.pyplot as plt

from .phasetracer import phase_tracer_info
from .plot import plot_temperature_uncertainty


def savejson(report, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)


def prepare_results_folder(folder=None):
    if folder is None:
        timestr = time.strftime("%Y%m%d_%H%M%S")
        folder = Path(f"transition_solver_results_{timestr}")
        os.mkdir(folder)
        return folder

    folder = Path(folder)

    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

    return folder


def save_transition_outputs(tr_report, tr_fig, phase_structure_raw, ctx, folder=None):
    """Save outputs available immediately after transition analysis."""
    folder = prepare_results_folder(folder)

    savejson(ctx.params, folder / "cli.json")
    savejson(phase_tracer_info(), folder / "phasetracer.json")
    tr_fig.savefig(folder / "tr.pdf")
    savejson(tr_report, folder / "tr.json")

    with open(folder / "phasetracer.txt", "w") as f:
        f.write(phase_structure_raw)

    return str(folder)


def save_gw_outputs(
    tr_report,
    gw_fig,
    analyser,
    detectors,
    folder,
    temperature_uncertainty=False,
):
    """Save outputs that require successful GW analysis."""
    folder = prepare_results_folder(folder)
    path_dirs = []

    gw_fig.savefig(folder / "gw.pdf")

    uncertainty_reports = {}
    uncertainty_figures = {}
    if temperature_uncertainty:
        transition_ids = list(
            dict.fromkeys(
                transition_id
                for path in tr_report["paths"]
                if path["valid"]
                for transition_id in path["transitions"]
            )
        )
        uncertainty_reports = (
            analyser.temperature_uncertainty_report_for_transition_ids(
                transition_ids, *detectors
            )
        )
        uncertainty_figures = {
            transition_id: plot_temperature_uncertainty(report, transition_id)
            for transition_id, report in uncertainty_reports.items()
        }

    # save results from each path
    for idx, path in enumerate(tr_report["paths"]):
        if not path["valid"]:
            continue

        # phases
        phases = "-".join(str(p) for p in path["phases"]) if path["phases"] else "none"

        # transitions
        transitions = "-".join(path["transitions"]) if path["transitions"] else "none"

        # folder name
        path_dir = folder / f"path_{idx}_p{phases}_t{transitions}"
        path_dir.mkdir(parents=True, exist_ok=True)
        path_dirs.append(
            {
                "index": idx,
                "phases": path["phases"],
                "transitions": path["transitions"],
                "directory": str(path_dir),
            }
        )

        path_gw_report = analyser.report_for_transition_ids(path["transitions"], *detectors)

        savejson(path_gw_report, path_dir / "gw.json")
        if temperature_uncertainty:
            path_uncertainty_reports = {
                transition_id: uncertainty_reports[transition_id]
                for transition_id in path["transitions"]
            }
            savejson(
                path_uncertainty_reports,
                path_dir / "gw_temperature_uncertainty.json",
            )
            for transition_id in path["transitions"]:
                uncertainty_figures[transition_id].savefig(
                    path_dir
                    / f"gw_temperature_uncertainty_transition_{transition_id}.pdf"
                )
        savejson(path, path_dir / "tr_path.json")

    for figure in uncertainty_figures.values():
        plt.close(figure)

    return str(folder), path_dirs
