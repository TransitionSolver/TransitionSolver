"""
Efficiency formula
==================
"""

import numpy as np
from scipy.integrate import odeint

try:
    from scipy.integrate import simps
except ImportError:
    from scipy.integrate import simpson as simps


def mu(a, b):
    """
    Based on Appendix B of https://arxiv.org/abs/2010.09744
    """
    return (a - b) / (1. - a * b)


def dfdv(xiw, v, cs2):
    """
    Based on Appendix B of https://arxiv.org/abs/2010.09744
    """
    xi, w = xiw
    dxidv = mu(xi, v)**2 / cs2 - 1.
    dxidv *= (1. - v * xi) * xi / 2. / v / (1. - v**2)
    dwdv = (1. + 1. / cs2) * mu(xi, v) * w / (1. - v**2)
    return [dxidv, dwdv]


def kappa_nu_model(cs2, al, vp, n=None):
    """
    Based on code from Appendix A of https://arxiv.org/abs/2004.06995
    """
    if n is None:
        n = int(1 + 1e6)
    
    nu = 1. / cs2 + 1.
    tmp = 1. - 3. * al + vp**2 * (1. / cs2 + 3. * al)
    disc = 4 * vp**2 * (1. - nu) + tmp**2
    
    if disc < 0:
        raise RuntimeError("vp too small for detonation")

    vm = (tmp + np.sqrt(disc)) / 2 / (nu - 1.) / vp
    wm = (-1. + 3. * al + (vp / vm) * (-1. + nu + 3. * al))
    wm /= (-1. + nu - vp / vm)

    def dfdv(xiw, v, nu):
        xi, w = xiw
        dxidv = ((xi - v) / (1. - xi * v))**2 * (nu - 1.) - 1.
        dxidv *= (1. - v * xi) * xi / 2. / v / (1. - v**2)
        dwdv = nu * (xi - v) / (1. - xi * v) * w / (1. - v**2)
        return [dxidv, dwdv]

    vs = np.linspace((vp - vm) / (1. - vp * vm), 0, n)
    sol = odeint(dfdv, [vp, 1.], vs, args=(nu,))
    xis, ws = (sol[:, 0], -sol[:, 1] * wm / al * 4. / vp**3)

    return simps(ws * (xis * vs)**2 / (1. - vs**2), xis)
