"""
Compute geff
============

Using digitised curves for fermions and bosons from top panels 
of Fig. 3 from https://arxiv.org/abs/1609.04979
"""

import os
from pathlib import Path

from scipy.interpolate import CubicSpline
import numpy as np

from .. import eigen


THIS = Path(os.path.dirname(os.path.abspath(__file__)))


def geff_from_disk(file_name):
    """
    @returns Interpolated geff from file on disk
    """
    t_over_m, geff = np.loadtxt(file_name, delimiter=",", unpack=True)
    geff[geff < 0.] = 0.  # geff cannot be negative

    order = np.argsort(t_over_m)
    t_over_m = t_over_m[order]
    geff = geff[order]

    interpolator = CubicSpline(t_over_m, geff)

    def clamped(t):
        """
        @returns Cubic-split with no extrapolation
        """
        result = interpolator(t)
        result[t > t_over_m.max()] = geff[np.argmax(t_over_m)]
        result[t < t_over_m.min()] = geff[np.argmin(t_over_m)]
        return result

    return clamped


boson_geff = geff_from_disk(THIS / 'boson_geff.csv')
fermion_geff = geff_from_disk(THIS / 'fermion_geff.csv')


def field_dependent_dof(potential, phi, T):
    """
    @param potential One-loop potential object
    @returns Degrees of freedom
    """
    dof = potential.raddof
    phi = eigen.vector(phi)

    # scalar and vector

    m2s = list(potential.get_scalar_masses_sq(phi, T))
    ns = list(potential.get_scalar_dofs())

    m2v = list(potential.get_vector_masses_sq(phi))
    nv = list(potential.get_vector_dofs())

    m2b = m2s + m2v
    nb = ns + nv

    for d, m2 in zip(nb, m2b):
        y = T / np.sqrt(np.abs(m2))
        dof += d * boson_geff(y)

    # fermion

    m2f = potential.get_fermion_masses_sq(phi)
    nf = potential.get_fermion_dofs()

    for d, m2 in zip(nf, m2f):
        y = T / np.sqrt(np.abs(m2))
        dof += d * fermion_geff(y)

    return dof
