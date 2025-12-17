"""
Efficiency formula
==================

Based on https://arxiv.org/abs/2004.06995
"""

import numpy as np
from scipy.integrate import solve_ivp, quad


def mu(xi, v):
    """
    @returns Eq. 16 for boosted fluid velocity
    """
    return (xi - v) / (1. - xi * v)


def gamma(v):
    """
    @returns Lorentz factor below eq. 5
    """
    return (1. - v**2)**-0.5


def kappa_nu_model(cs2, al, vp, use_cj=True):
    r"""
    Based on code from Appendix A of https://arxiv.org/abs/2004.06995
    
    @param cs2  $c_s^2$, the speed of sound squared
    @param al $\alpha_\bar\theta$, the strength of PT
    @param vp $v_p \equiv \xi_w$, the bubble wall velocity
    
    @returns $\kappa_{\bar\theta}$, the efficiency
    """
    nu = 1. / cs2 + 1.
    
    if not use_cj:
        # eq. 52
        c = 1. - 3. * al + vp**2 * (1. / cs2 + 3. * al)
        # eq. 53
        d = c**2 - 4 * vp**2 / cs2

        if d < 0:
            raise RuntimeError("vp too small for detonation")

        # eq. 51 for fluid velocity behind wall
        vm = cs2 * (c + np.sqrt(d)) / (2 * vp)
    else:
        # analytic result in if using Chapman-Jouget velocity
        vm = np.sqrt(cs2)

    def dxi_dv(v, xi):
        r"""
        @returns Eq. 14
        """
        return 0.5 * gamma(v)**2 * (1. - v * xi) * (mu(xi, v)**2 / cs2 - 1.) * xi / v

    def dw_dv(v, xi, w):
        r"""
        @returns Eq. 15
        """
        return nu * gamma(v)**2 * mu(xi, v) * w

    def dxi_w_dv(v, xi_w):
        r"""
        @returns Eq. 14 and 15
        """
        xi, w= xi_w
        return [dxi_dv(v, xi), dw_dv(v, xi, w)]

    # eq. 54 for enthalpy
    wm = (vp / vm) * (1. / cs2 + 3. * al) - 1. + 3. * al 
    wp = 1. / cs2 - vp / vm

    # boundary condition for \xi
    xi0 = vp
    # boundary condition for w
    w0 = wm
    # boundaries for v
    v0 = mu(vp, vm)
    v1 = 0.
    # solve system of differential equations for \xi and w
    sol = solve_ivp(dxi_w_dv, [v0, v1], [xi0, w0], vectorized=True, dense_output=True)

    def rho_fl_integrand(v):
        """
        Note that integrating wrt v hence dxidv factor
        
        @returns Eq. 5 integrand
        """
        xi, w = sol.sol(v)
        return dxi_dv(v, xi) * xi**2 * v**2 * gamma(v)**2 * w

    # eq. 5
    rho_fl = 3. / vp**3 * quad(rho_fl_integrand, v1, v0)[0]

    # eq. 34
    D_theta = 3. * al * wp

    # eq. 36 for kappa
    return 4 * rho_fl / D_theta
