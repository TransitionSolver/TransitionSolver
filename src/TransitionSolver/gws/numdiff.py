"""
Implementation of numerical derivatives
=======================================
"""

import warnings
import numpy as np


def approx_zero(x):
    return np.abs(x) <= np.finfo(float).tiny


def derivatives(f, x, dx, order_4=True):
    """
    @returns Function and first and second derivatives
    """
    f0 = f(x)
    f1 = f(x + dx)
    fm1 = f(x - dx)
    dx2 = dx * dx

    if approx_zero(dx) or approx_zero(dx2):
        warnings.warn("Stepsize was zero or subnormal; check for underflow")

    if not order_4:
        dfdx = (f1 - fm1) / (2 * dx)
        d2fdx2 = (f1 - 2 * f0 + fm1) / dx2
        if approx_zero(dfdx) or approx_zero(d2fdx2):
            warnings.warn("Derivatives were zero or subnormal; check for underflow")
        return f0, dfdx, d2fdx2

    f2 = f(x + 2 * dx)
    fm2 = f(x - 2 * dx)

    dfdx = (-f2 + 8 * f1 - 8 * fm1 + fm2) / (12 * dx)
    d2fdx2 = (-f2 + 16 * f1 - 30 * f0 + 16 * fm1 - fm2) / (12 * dx2)

    if approx_zero(dfdx) or approx_zero(d2fdx2):
        warnings.warn("Derivatives were zero or subnormal; check for underflow")

    return f0, dfdx, d2fdx2
