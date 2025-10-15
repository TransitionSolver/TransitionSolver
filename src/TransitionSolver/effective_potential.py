"""
Access effective potentials from PhaseTracer
============================================
"""

import os
from pathlib import Path

import cppyy
import numpy as np
from numpy import pi

from . import eigen


EP_HOME = Path(os.getenv("PHASETRACER", Path.home() / ".PhaseTracer")) / "EffectivePotential"
EP_INCLUDE = EP_HOME / "include" / "effectivepotential"
EP_MODELS = EP_HOME / "include" / "models"
EP_LIB = EP_HOME / "lib" / "libeffectivepotential.so"


class MixinPotential:
    """
    Adapt raw C++ potential for Pythonic access.

    E.g., accept/return np.array arguments instead of eigen3
    """

    def __call__(self, phi, T):
        """
        @returns Total potential
        """
        if phi.ndim > 1:
            return np.array([self(p, T) for p in phi])
        return self.V(eigen.vector(phi), T)

    def grad(self, phi, T):
        """
        @returns Gradient of total potential
        """
        if phi.ndim > 1:
            return np.array([self.grad(p, T) for p in phi])
        return eigen.to_numpy(self.dV_dx(eigen.vector(phi), T))

    def free_energy_density(self, X, T):
        """
        @returns Free energy density by subtracting radiation contributions
        """
        self.raddof = 80.25  # TODO this is real scalar singlet specific
        return self(X, T) - pi**2 / 90 * self.raddof * T**4

    @property
    def minimum_temperature(self):
        return 0.1  # TODO what is this

    def dof_in_phase(self, *args, **kwargs):
        return 106.75  # TODO this is real scalar singlet specific and an approximation


def load_potential(params, header_file, class_name=None, lib_name=None):
    """
    @param params Parameters for constructor e.g., Lagrangian parameters
    @param header_file Header file where model defined
    @param class_name Name of model in header file
    @param lib_name If model compiled (rather than header only), name of built file

    @returns Potential object
    """
    if class_name is None:
        class_name = str(Path(header_file).stem)

    # include headers

    for pth in [EP_INCLUDE, EP_MODELS]:
        cppyy.add_include_path(str(pth))

    cppyy.include(header_file)

    # load libraries

    cppyy.load_library(str(EP_LIB))

    if lib_name is not None:
        cppyy.load_library(lib_name)

    # make potential

    Potential = getattr(cppyy.gbl.EffectivePotential, class_name)

    class ExtendedPotential(MixinPotential, Potential):
        pass

    return ExtendedPotential(*params)
