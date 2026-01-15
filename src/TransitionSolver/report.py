"""
Make a report of results
========================
"""

import time
import os
from pathlib import Path

import json_numpy as json

from . import phase_tracer_info


def savejson(report, file_name):
    with open(file_name, "w") as f:
        json.dump(report, f)


def make_results_folder():
    timestr = time.strftime("%Y%m%d_%H%M%S")
    folder = Path(f"transition_solver_results_{timestr}")
    os.mkdir(folder)
    return folder


def saveall(tr_report, gw_report, tr_fig, gw_fig, ctx):
    """
    Save all results to a folder named by date/time

    @returns Name of results folder
    """
    folder = make_results_folder()
    savejson(ctx.params, folder / "cli.json")
    savejson(phase_tracer_info(), folder / "phasetracer.json")
    gw_fig.savefig(folder / "gw.pdf")
    tr_fig.savefig(folder / "tr.pdf")
    savejson(gw_report, folder / "gw.json")
    savejson(tr_report, folder / "tr.json")
    return str(folder)
