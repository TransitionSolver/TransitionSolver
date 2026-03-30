"""
Make a report of results
========================
"""

import time
import os
from pathlib import Path

import json

from . import phase_tracer_info


def savejson(report, file_name):
    with open(file_name, "w") as f:
        json.dump(report, f, indent=4)


def make_results_folder():
    timestr = time.strftime("%Y%m%d_%H%M%S")
    folder = Path(f"transition_solver_results_{timestr}")
    os.mkdir(folder)
    return folder


def saveall(tr_report, gw_report, tr_fig, gw_fig, phase_structure_raw, ctx, analyser, detectors, folder=None):
    if folder is None:
        folder = make_results_folder()

    folder = Path(folder)

    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

    savejson(ctx.params, folder / "cli.json")
    savejson(phase_tracer_info(), folder / "phasetracer.json")
    gw_fig.savefig(folder / "gw.pdf")
    tr_fig.savefig(folder / "tr.pdf")
    savejson(tr_report, folder / "tr.json")

    with open(folder / "phasetracer.txt", "w") as f:
        f.write(phase_structure_raw)

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

        selected = set(path["transitions"])
        path_gw_report = analyser.report_for_transition_ids( path["transitions"],*detectors)

        savejson(path_gw_report, path_dir / "gw.json")
        savejson(path, path_dir / "tr_path.json")

    return str(folder)    
