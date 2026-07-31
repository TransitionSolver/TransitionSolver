"""
Make a report of results
========================
"""

import time
import os
from pathlib import Path

import json

from .phasetracer import phase_tracer_info


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


def save_gw_outputs(tr_report, gw_fig, analyser, detectors, folder):
    """Save outputs that require successful GW analysis."""
    folder = prepare_results_folder(folder)
    path_dirs = []

    gw_fig.savefig(folder / "gw.pdf")

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
        savejson(path, path_dir / "tr_path.json")

    return str(folder), path_dirs
