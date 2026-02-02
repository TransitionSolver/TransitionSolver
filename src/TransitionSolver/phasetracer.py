"""
PhaseTracer interface using system calls
========================================
"""

import os
import datetime
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .analysis.phase_structure import Phase, Transition, PhaseStructure


CWD = os.path.dirname(os.path.abspath(__file__))

DEFAULT = Path.home() / ".TransitionSolver" / "phasetracer-src"
PT_HOME = Path(os.getenv("PHASETRACER", DEFAULT))
PT_LIB = PT_HOME / "lib" / "libphasetracer.so"
PT_INCLUDE = PT_HOME / "include"
EP_HOME = PT_HOME / "EffectivePotential"
EP_INCLUDE = EP_HOME / "include" / "effectivepotential"
EP_MODELS = EP_HOME / "include" / "models"
EP_LIB = EP_HOME / "lib" / "libeffectivepotential.so"
CXX = "g++"
TEMPLATE_CPP = os.path.join(CWD, "interface.cpp")
LIBS = [EP_LIB, PT_LIB, "-lboost_log", "-lboost_filesystem", "-lnlopt"]
PT_UNIT_TEST = PT_HOME / "bin" / "unit_tests"
DEFAULT_NAMESPACE = ("EffectivePotential",)


def phase_tracer_info():
    """
    @returns Information about PhaseTracer installation
    """
    version = subprocess.check_output(["git", "describe", "--dirty", "--always"], cwd=PT_HOME, text=True)
    version = version.strip()
    build_time = os.path.getmtime(PT_LIB)
    build_time = str(datetime.datetime.fromtimestamp(build_time))
    return {"HOME": str(PT_HOME), "GIT": version, "BUILT": build_time}


def rpath(name):
    """
    @returns Compiler argument to add an rpath
    """
    return f"-Wl,-rpath={name}"


def build_phase_tracer(model_header, model=None, model_lib=None, model_namespace=DEFAULT_NAMESPACE, force=False):
    """
    Build PhaseTracer model for use in TransitionSolver

    @param model Name of model in C++ header
    @param model_header Header file where model defined
    @param model_lib Library for model, if not header-only
    @param model_namespace Any namespaces under which model appears in header, as list
    @param force Force recompilation even if executable already exists

    @returns Path to built executable
    """
    if model is None:
        model = str(Path(model_header).stem)

    exe_name = PT_HOME / model

    if os.path.exists(exe_name) and not force:
        return exe_name

    cmd = [CXX, TEMPLATE_CPP, "-o", exe_name, "-I", PT_INCLUDE, "-I", EP_MODELS, "-I", EP_INCLUDE, "-I", CWD, rpath(EP_HOME / 'lib'), rpath(PT_HOME / 'lib')] + LIBS

    if model_lib:
        cmd.append(model_lib)

    if model_namespace:
        joined = "::".join(model_namespace)
        model = f"{joined}::{model}"

    cmd.append(f"-DMODEL_NAME_WITH_NAMESPACE={model}")
    cmd.append(f'-DMODEL_HEADER="{model_header}"')

    compile_ = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if compile_.returncode != 0:
        raise RuntimeError(compile_.stderr)

    return exe_name

# should remno
def run_phase_tracer(exe_name, point_file=None, point=None,  pt_settings_file=None) -> str:
    """
    Run PhaseTracer and read serialised data
    @param exe_name Name of executable
    @param point_file File containing parameter point
    @param point Array containing parameter point
    @param pt_settings_file Optional JSON file of PhaseFinder/TransitionFinder overrides
    """
    if point_file is None:
        with tempfile.NamedTemporaryFile() as f:
            point = np.array(point).reshape(1, len(point))
            np.savetxt(f.name, point)
            return run_phase_tracer(exe_name, f.name, pt_settings_file=pt_settings_file)

    cmd = [str(exe_name), str(point_file)]
    if pt_settings_file is not None:
        cmd.append(str(pt_settings_file))

    run = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if run.returncode != 0:
        raise RuntimeError(run.stderr)
    return run.stdout

def read_path(data):
    """
    @returns Transition path from lines of data
    """
    return [int(el[1:]) - 1 if el.startswith('-') else int(el) for el in data[0].split()]


def read_arr(data):
    """
    @returns Array from lines of data
    """
    return np.squeeze(np.array([np.fromstring(d, sep=' ') for d in data]))


def read_phase_tracer(phase_tracer_data=None, phase_tracer_file=None) -> PhaseStructure:
    """
    Read serialised data from PhaseTracer

    @returns Phase structure object from PhaseTracer serialised data
    """
    if phase_tracer_file is not None:
        with open(phase_tracer_file, encoding="utf8") as f:
            phase_tracer_data = f.read()

    phases = []
    transitions = []
    paths = []

    parts = [part.split("\n") for part in phase_tracer_data.strip().split("\n\n")]

    for part in parts:

        if not part[0].startswith("#"):
            raise RuntimeError(f"Could not read {part}")

        metadata = part[0].lstrip("#").strip()
        arr = part[1:]

        if not arr:
            continue

        try:
            label, key = metadata.split()
        except ValueError:
            label = metadata

        if label == "phase":
            phases.append(Phase(key, read_arr(arr)))
        elif label == "transition":
            transitions.append(Transition(read_arr(arr)))
        elif label == "transition-path":
            paths.append(read_path(arr))
        else:
            raise RuntimeError(f"Could not read {part}")

    return PhaseStructure(phases, transitions, paths)
