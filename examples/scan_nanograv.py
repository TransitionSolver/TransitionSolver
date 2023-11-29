"""
Solve for kappa to find benchmarks
===================================
"""

import os
import functools
import sys

import numpy as np
from scipy.optimize import toms748

from models.Archil_model import SMplusCubic
from examples.command_line_interface import main
from examples.remake_nanograv_plots import get_report


def partial_class(cls, *args, **kwargs):
    """
    functools partial, but for class constructor
    """
    class PartialClass(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)
    return PartialClass


def run(kappa, folder_name, **kwargs):
    """
    Run a point with particular kappa
    
    :param folder_name: Results saved to this folder
    :returns: Results dictionary
    """
    potential_class = partial_class(SMplusCubic, bUseBoltzmannSuppression=True, **kwargs)

    # make result folder
    os.makedirs(folder_name, exist_ok=True)

    # get parameter point
    potential = potential_class(kappa)
    parameter_point = potential.getParameterPoint()

    # save to disk for future reference
    np.savetxt(os.path.join(folder_name, 'parameter_point.txt'), parameter_point)

    # run toolchain
    main(potential_class, folder_name, "run_supercool", ["-boltz"], parameter_point,
        bDebug=False, bPlot=False, bUseBoltzmannSuppression=True)

    return get_report(folder_name)


def solve_t_p(a, b, target_t_p, folder_name="solve_t_p", rtol=1e-6, xtol=1e-3, **kwargs):
    """
    Solve for kappa inside an interval with a particular T_p
    """
    def objective(kappa):

        data = run(kappa, folder_name, **kwargs)
        t_p = data["transitions"][0].get("Tp")

        if not t_p:
            delta = sys.float_info.max
        else:
            delta = np.log(target_t_p / t_p)

        print(f"kappa = {kappa}. T_p = {t_p}. Target = {target_t_p}. Delta = {delta}")

        return delta

    return toms748(objective, a, b, rtol=rtol, xtol=xtol, full_output=True)


def solve_completion(a, b, folder_name="solve_completion", rtol=1e-6, xtol=1e-3, **kwargs):
    """
    Solve for kappa inside an interval such false volume starts decreasing exactly at T_p
    """
    def objective(kappa):

        data = run(kappa, folder_name, **kwargs)
        t_p = data["transitions"][0].get("Tp")
        t_decreasing = data["transitions"][0].get("TVphysDecr_high", 0.)

        if not t_p:
            delta = sys.float_info.max
        else:
            delta = t_p - t_decreasing

        print(f"kappa = {kappa}. t_p = {t_p }. t_decreasing = {t_decreasing}. Delta = {delta}")

        return delta

    return toms748(objective, a, b, rtol=rtol, xtol=xtol, full_output=True)


if __name__ == "__main__":

    # BP1

    res = solve_completion(-118., -117.)
    print(res)
    run(res[0], "BP1")

    # BP2

    res = solve_t_p(-119., -118., 0.1, rtol=1e-8, xtol=1e-6)
    print(res)
    run(res[0], "BP2")

    # look at changing higgs mass for BP1

    delta_mh = 0.17

    res = solve_completion(-118.5, -117.5, mh=125 + delta_mh)
    print(res)
    run(res[0], "BP1_increase_mh")

    res = solve_completion(-118.5, -117.5, mh=125 - delta_mh)
    print(res)
    run(res[0], "BP1_decrease_mh")
