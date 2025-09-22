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


def kappa_nu_model(cs2, al, vp, use_cj=True, n=None):
    """
    Based on code from Appendix A of https://arxiv.org/abs/2004.06995
    """
    if n is None:
        n = int(1 + 1e6)
    
    nu = 1. / cs2 + 1.
    
    if not use_cj:
        tmp = 1. - 3. * al + vp**2 * (1. / cs2 + 3. * al)
        disc = 4 * v**2 * (1 - nu) + tmp**2

        if disc < 0:
            raise RuntimeError("vp too small for detonation")

        vm = (tmp + np.sqrt(disc)) / 2 / (nu - 1.) / vp
    else:
        vm = cs2**0.5
        
    wm = (-1. + 3. * al + (vp / vm) * (-1. + nu + 3. * al))
    wm /= (-1. + nu - vp / vm)

    def dfdv(xiw, v):
        xi, w = xiw
        dxidv = ((xi - v) / (1. - xi * v))**2 * (nu - 1.) - 1.
        dxidv *= (1. - v * xi) * xi / 2. / v / (1. - v**2)
        dwdv = nu * (xi - v) / (1. - xi * v) * w / (1. - v**2)
        return [dxidv, dwdv]

    vs = np.linspace((vp - vm) / (1. - vp * vm), 0, n)
    sol = odeint(dfdv, [vp, 1.], vs)
    xis, ws = (sol[:, 0], -sol[:, 1] * wm / al * 4. / vp**3)

    return simps(ws * (xis * vs)**2 / (1. - vs**2), xis)
