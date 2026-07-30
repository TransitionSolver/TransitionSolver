"""
Make a report of results
========================
"""

import time
import os
import shutil
from pathlib import Path

import json
import matplotlib.pyplot as plt

from .phasetracer import phase_tracer_info
from .plot import plot_temperature_scan


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


def save_transition_outputs(
    tr_report,
    tr_fig,
    phase_structure_raw,
    ctx,
    folder=None,
    point_file_name=None,
):
    """Save outputs available immediately after transition analysis."""
    folder = prepare_results_folder(folder)

    savejson(ctx.params, folder / "cli.json")
    savejson(phase_tracer_info(), folder / "phasetracer.json")
    tr_fig.savefig(folder / "tr.pdf")
    savejson(tr_report, folder / "tr.json")

    with open(folder / "phasetracer.txt", "w") as f:
        f.write(phase_structure_raw)

    if point_file_name is not None:
        saved_point = folder / "parameter_point.txt"
        if Path(point_file_name).resolve() != saved_point.resolve():
            shutil.copyfile(point_file_name, saved_point)

    return str(folder)


def save_gw_outputs(
    tr_report,
    gw_fig,
    analyser,
    detectors,
    folder,
    temperature_scan=False,
    temperature_uncertainty=False,
    ptas=None,
    additional_transition_ids=(),
    transition_diagnostics=None,
):
    """Save outputs that require successful GW analysis."""
    folder = prepare_results_folder(folder)
    path_dirs = []
    ptas = ptas or []
    transition_diagnostics = transition_diagnostics or {}

    gw_fig.savefig(folder / "gw.pdf")

    valid_transition_ids = list(
        dict.fromkeys(
            transition_id
            for path in tr_report["paths"]
            if path["valid"]
            for transition_id in path["transitions"]
        )
    )
    additional_transition_ids = [
        str(transition_id)
        for transition_id in additional_transition_ids
        if str(transition_id) not in valid_transition_ids
    ]

    transition_ids = valid_transition_ids + additional_transition_ids
    uncertainty_transition_ids = [
        transition_id
        for transition_id in transition_ids
        if analyser.transition_reports[transition_id].get("T_p") is not None
        and analyser.transition_reports[transition_id].get("T_f") is not None
    ]

    scan_reports = {}
    scan_plot_files = {}
    if temperature_scan:
        scan_reports = analyser.temperature_scan_report_for_transition_ids(
            transition_ids, *detectors
        )

    if temperature_scan:
        for transition_id, report in scan_reports.items():
            figure = plot_temperature_scan(report, transition_id)
            plot_file = folder / f"gw_temperature_scan_transition_{transition_id}.pdf"
            figure.savefig(plot_file)
            plt.close(figure)
            scan_plot_files[transition_id] = plot_file

    uncertainty_reports = {}
    if temperature_uncertainty:
        uncertainty_reports = (
            analyser.temperature_uncertainty_report_for_transition_ids(
                uncertainty_transition_ids,
                *detectors,
                scan_reports=scan_reports,
            )
        )

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
        for transition_id in path["transitions"]:
            if transition_id in transition_diagnostics:
                path_gw_report[transition_id]["Transition diagnostics"] = (
                    transition_diagnostics[transition_id]
                )

        savejson(path_gw_report, path_dir / "gw.json")
        if temperature_scan:
            path_scan_reports = {
                transition_id: scan_reports[transition_id]
                for transition_id in path["transitions"]
                if transition_id in scan_reports
            }
            savejson(
                path_scan_reports,
                path_dir / "gw_temperature_scan.json",
            )
            for transition_id in path_scan_reports:
                shutil.copyfile(
                    scan_plot_files[transition_id],
                    path_dir / f"gw_temperature_scan_transition_{transition_id}.pdf",
                )

        if temperature_uncertainty:
            path_uncertainty_reports = {
                transition_id: uncertainty_reports[transition_id]
                for transition_id in path["transitions"]
                if transition_id in uncertainty_reports
            }
            savejson(
                path_uncertainty_reports,
                path_dir / "gw_temperature_uncertainty.json",
            )
        savejson(path, path_dir / "tr_path.json")

    for transition_id in additional_transition_ids:
        transition_dir = (
            folder / "transitions_with_perc_temp" / f"transition_{transition_id}"
        )
        transition_dir.mkdir(parents=True, exist_ok=True)

        report = analyser.gws[transition_id].report(*detectors)
        if transition_id in transition_diagnostics:
            report["Transition diagnostics"] = transition_diagnostics[transition_id]
        savejson(report, transition_dir / "gw.json")

        figure = analyser.plot_for_transition_ids(
            [transition_id], detectors=detectors, ptas=ptas
        )
        figure.savefig(transition_dir / "gw.pdf")
        plt.close(figure)

        if temperature_scan:
            savejson(
                {transition_id: scan_reports[transition_id]},
                transition_dir / "gw_temperature_scan.json",
            )
            shutil.copyfile(
                scan_plot_files[transition_id],
                transition_dir / f"gw_temperature_scan_transition_{transition_id}.pdf",
            )

        if temperature_uncertainty:
            report = (
                {transition_id: uncertainty_reports[transition_id]}
                if transition_id in uncertainty_reports
                else {}
            )
            savejson(
                report,
                transition_dir / "gw_temperature_uncertainty.json",
            )

    return str(folder), path_dirs
