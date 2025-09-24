"""
Efficiency formula
==================
"""

import numpy as np
from scipy.integrate import solve_ivp, quad


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
        
    def dfdv(v, xiw):
        xi, w = xiw
        # below eq. 5 - Lorentz factor
        gamma_2 = 1. / (1. - v**2)
        # eq. 16 for boosted fluid velocity
        mu = (xi - v) / (1. - xi * v)
        # eq. 14
        dxidv = 0.5 * gamma_2 * (1. - v * xi) * (mu**2 / cs2 - 1.) * xi / v
        # eq. 15
        dwdv = nu * gamma_2 * mu * w 
        return [dxidv, dwdv]

    y0 = [vp, 1.]  # boundary condition
    t_span = [(vp - vm) / (1. - vp * vm), 0]
    sol = solve_ivp(dfdv, t_span, y0, vectorized=True, dense_output=True)
    
    def integrand(v):
        xi, w = sol.sol(v)
        dxidv = dfdv(v, [xi, w])[0]
        # below eq. 5 - Lorentz factor
        gamma_2 = 1. / (1. - v**2)
        # eq. 5 integrand, though integrating wrt v hence dxidv
        return dxidv * xi**2 * v**2 * gamma_2 * w
        
    # eq. 5
    rho_fl = 3. / vp**3 * quad(integrand, t_span[1], t_span[0])[0]
    
    # eq. 54 for enthalpy ratio
    wm = (vp / vm) * (1. / cs2 + 3. * al) - 1. + 3. * al 
    wp = 1. / cs2 - vp / vm
    r = wp / wm
    
    # eq. 34
    D_theta = 3. * al * r  # TODO this doesn't match the eq. 34
    
    # eq. 36 for kappa
    return 4 * rho_fl / D_theta
