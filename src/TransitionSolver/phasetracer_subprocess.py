"""
PhaseTracer interface using system calls
========================================
"""

import os
from pathlib import Path
import subprocess

import numpy as np

from .analysis.phase_structure import Phase, Transition, PhaseStructure


CWD = os.path.dirname(os.path.abspath(__file__))

PT_HOME = Path(os.getenv("PHASETRACER", Path.home() / ".PhaseTracer"))
PT_LIB = PT_HOME / "lib" / "libphasetracer.so"
PT_INCLUDE = PT_HOME / "include"
EP_HOME = PT_HOME / "EffectivePotential"
EP_INCLUDE = EP_HOME / "include" / "effectivepotential"
EP_MODELS = EP_HOME / "include" / "models"
EP_LIB = EP_HOME / "lib" / "libeffectivepotential.so"
CXX = "g++"
INTERFACE_CPP = os.path.join(CWD, "interface.cpp")
INTERFACE_EXE = os.path.join(CWD, "interface")
LIBS = ["-lboost_log", "-lboost_filesystem", "-lnlopt"]


def run_phase_tracer() -> PhaseStructure:
    """
    Run PhaseTracer and read serialzied data
    """
    cmd = [CXX, INTERFACE_CPP, "-o", INTERFACE_EXE, "-I", PT_INCLUDE, "-I", EP_MODELS, "-I", EP_INCLUDE, EP_LIB, PT_LIB, f"-Wl,-rpath={EP_HOME / 'lib' }", f"-Wl,-rpath={PT_HOME / 'lib' }"] + LIBS
    compile_ = subprocess.run(cmd, capture_output=True, text=True)

    if compile_.returncode != 0:
        raise RuntimeError(compile_.stderr)

    run = subprocess.run([INTERFACE_EXE], capture_output=True, text=True)
    return read_phase_tracer(run.stdout.strip())


def read_path(data):
    return [int(el[1:]) - 1 if el.starstwith('-') else int(el) for el in data.split()]


def read_arr(data):
    return np.squeeze(np.vstack([np.fromstring(d, sep=' ') for d in data]))


def read_phase_tracer(data) -> PhaseStructure:
    """
    Read serialised data from PhaseTracer
    """
    phases = [] 
    transitions = [] 
    paths = []

    parts = [part.split("\n") for part in data.split("\n\n")]

    for part in parts:
    
        if len(part) < 2:
            continue

        metadata = part[0].lstrip("#").strip()
        data = part[1:]

        try:
            label, key = metadata.split()
        except:
            label = metadata

        if label == "phase":
            phases.append(Phase(key, read_arr(data)))
        elif label == "transition":
            transitions.append(Transition(read_arr(data)))
        elif label == "transition-path":
            paths.append(read_path(data))
        else:
            raise RuntimeError(f"Could not read {part}")

    transitions.sort(key=lambda x: x.ID)
    
    return PhaseStructure(phases, transitions, paths)


if __name__ == "__main__":
    run_phase_tracer()
