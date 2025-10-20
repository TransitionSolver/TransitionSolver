"""
Compute g_eff
==============

Currently using digitised curves from top panels 
of Fig. 3 from https://arxiv.org/abs/1609.04979
"""

import os
from pathlib import Path

from scipy.interpolate import CubicSpline
import numpy as np

from TransitionSolver import eigen  # TODO rel import


THIS = Path(os.path.dirname(os.path.abspath(__file__)))


def geff_from_disk(file_name):
    """
    @returns Interpolated g_eff from file on disk
    """
    x, y = np.loadtxt(file_name, delimiter=",", unpack=True)
    y[y < 0.] = 0.  # TODO why not allow negative, it's in data files

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    interpolator = CubicSpline(x, y)

    def clamped(x):
        result = interpolator(x)
        result[x > x.max()] = y[np.argmax(x)]
        result[x < x.min()] = y[np.argmin(x)]
        return result

    return clamped


boson_geff = geff_from_disk(THIS / ".." / 'data' / 'boson_geff.csv')
fermion_geff = geff_from_disk(THIS / ".." / 'data' / 'fermion_geff.csv')


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
