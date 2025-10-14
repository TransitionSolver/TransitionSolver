"""
Access effective potentials from PhaseTracer
============================================
"""

from pathlib import Path

import cppyy
import numpy as np


EP_HOME = Path(os.getenv("PHASETRACER", Path.home() / ".PhaseTracer")) / "EffectivePotential"
EP_INCLUDE = EP_HOME / "include" / "effectivepotential"
EP_MODELS = EP_HOME / "include" / "models"
EP_LIB = EP_HOME / "lib" / "libeffectivepotential.so"


cppyy.load_library(str(EP_LIB))

for pth in [EP_INCLUDE, EP_MODELS]:
    cppyy.add_include_path(str(pth))


def load_potential(params, header_file, class_name=None, lib_name=None):
    """
    @param params Parameters for constructor e.g., Lagrangian parameters
    @param header_file Header file where model defined
    @param class_name Name of model in header file
    @param lib_name If model compiled (rather than header only), name of built file

    @returns I
    """
    if class_name is None:
        class_name = Path(header_file).stem

    cppyy.include(header_file)

    if lib_name is not None:
        cppyy.load_library(lib_name)

    from cppyy.gbl import EffectivePotential
    Potential = getattr(EffectivePotential, class_name)
    return Potential(*params)


point = 'rss_bp1.txt'
potential = load_potential(np.loadtxt(point), "RSS.hpp")
print(potential(1, 2))
