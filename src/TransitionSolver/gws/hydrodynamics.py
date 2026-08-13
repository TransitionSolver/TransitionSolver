"""
Hydrodynamic quantities for phase transitions and gravitational waves
=====================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from numpy import pi

from ..analysis.phase_structure import Phase
from .numdiff import derivatives


GRAV_CONST = 6.7088e-39


def hubble_squared_from_energy_density(energy_density):
    return 8 * pi * GRAV_CONST / 3 * energy_density


@dataclass
class HydroVars:
    """
    Represent hydrodynamic variables in true and false vacuum
    """
    pressureFalse: float
    energyDensityFalse: float
    enthalpyDensityFalse: float
    entropyDensityFalse: float
    soundSpeedSqFalse: float

    pressureTrue: float
    energyDensityTrue: float
    enthalpyDensityTrue: float
    entropyDensityTrue: float
    soundSpeedSqTrue: float

    T: float
    cosmologicalEnergyDensityFalse: float
    assumesRadiationDomination: bool

    @property
    def traceAnomalyFalse(self):
        return (self.energyDensityFalse - 3 * self.pressureFalse) / 4

    @property
    def traceAnomalyTrue(self):
        return (self.energyDensityTrue - 3 * self.pressureTrue) / 4

    @property
    def pseudotraceFalse(self):
        return (self.energyDensityFalse -
                self.pressureFalse / self.soundSpeedSqTrue) / 4

    @property
    def pseudotraceTrue(self):
        # note following #arXiv:2004.06995 we use the speed of sound
        # in the *true* vacuum here
        return (self.energyDensityTrue -
                self.pressureTrue / self.soundSpeedSqTrue) / 4

    @property
    def alpha(self):
        return 4 / 3 * (self.pseudotraceFalse -
                        self.pseudotraceTrue) / self.enthalpyDensityFalse

    @property
    def soundSpeedTrue(self):
        return self.soundSpeedSqTrue**0.5

    @property
    def soundSpeedFalse(self):
        return self.soundSpeedSqFalse**0.5

    @property
    def cj_velocity(self):
        """
        @returns Chapman-Jouguet velocity
        """
        return (1. + (3. * self.alpha * (1. - self.soundSpeedSqTrue + 3. * self.soundSpeedSqTrue *
                self.alpha))**0.5) / (1. / self.soundSpeedTrue + 3. * self.soundSpeedTrue * self.alpha)

    @property
    def available_energy_fraction(self):
        """
        @returns fraction of energy available for sourcing GWs before
        accounting for the efficiency coefficients to specific sources
        (ie taking kappa = 1)
        """
        return (self.pseudotraceFalse - self.pseudotraceTrue) / \
            self.cosmologicalEnergyDensityFalse

    @property
    def hubble_constant(self):
        return hubble_squared_from_energy_density(
            self.cosmologicalEnergyDensityFalse
        )**0.5

    def average_pressure_density(self, pf):
        """
        """
        pt = 1 - pf
        return self.pressureFalse * pf + self.pressureTrue * pt

    def adiabatic_index(self, pf):
        """
        Use energy conservation for the average energy and average the phase
        pressures when the ground-state normalisation is known. When assuming
        radiation-domination, as currently required when the phases cannot
        extend to T=0, use Gamma = 4/3.
        """
        if self.assumesRadiationDomination:
            return 4 / 3
        return 1. + self.average_pressure_density(pf) / self.energyDensityFalse


def interpolate_hydro_vars(
        hv1: HydroVars, hv2: HydroVars, T: float) -> HydroVars:
    """
    @returns Hydrodynamic variables from linear interpolation between two temperatures
    """
    def linear_interpolate(val1: float, val2: float) -> float:
        f = (T - hv1.T) / (hv2.T - hv1.T)
        return val1 + f * (val2 - val1)

    if hv1.assumesRadiationDomination != hv2.assumesRadiationDomination:
        raise ValueError(
            "Cannot interpolate between different cosmological assumptions"
        )

    data = [
        linear_interpolate(
            getattr(
                hv1, f.name), getattr(
                hv2, f.name)) for f in fields(HydroVars)
        if f.name != "assumesRadiationDomination"]
    return HydroVars(*data, hv1.assumesRadiationDomination)


def guess_delta_t(from_phase: Phase, to_phase: Phase, potential, T: float) -> float:
    """
    @returns Delta T that wouldn't take us below Tmin or above Tmax

    N.B. we make 2 steps f(x +/- 2 * dx) to compute second derivatives hence
    factors of 0.5 below.

    We don't care about Tc because we can sample in the region
    Tc < T < Tmax for the purpose of differentiation
    """
    Tmin = max(from_phase.T[0], to_phase.T[0])
    Tmax = min(from_phase.T[-1], to_phase.T[-1])
    delta_t = max(0.0005 * Tmax, 0.0001 * potential.get_temperature_scale())
    return min(delta_t, 0.5 * (T - Tmin), 0.5 * (Tmax - T))


def _make_hydro_vars(phase, potential, T, delta_t, ground_state_energy=0.):

    def free_energy(x: float) -> float:
        phi = phase.find_phase_at_t(x, potential)
        return potential.free_energy_density(phi, x) - ground_state_energy

    f, df, d2f = derivatives(free_energy, T, delta_t)

    # Pressure
    p = -f

    # Energy density
    e = f - T * df

    # Enthalpy density
    w = -T * df

    # Entropy density
    s = -df

    # Sound speed squared
    c2 = df / (T * d2f)

    return {"p": p, "e": e, "w": w, "s": s, "c2": c2}


def radiation_energy_density(potential, T: float) -> float:
    raddof = potential.raddof
    return pi**2 / 30 * raddof * T**4


def _cosmological_energy_density(
        hvf, hvt, potential, T: float, ground_state_energy) -> float:
    if ground_state_energy is not None:
        return hvf["e"]

    # When the potential is unavailable at T = 0, approximate
    # rho_true(T) - rho_ground(0) by the radiation energy density. Thus
    # rho_false(T) - rho_ground(0) = Delta rho(T) + rho_r(T).
    return radiation_energy_density(potential, T) \
        + hvf["e"] - hvt["e"]


def make_hydro_vars(from_phase: Phase, to_phase: Phase, potential, T: float,
                    ground_state_energy=0.) -> HydroVars:
    """
    @returns Hydrodynamical variables in true and false vacuum
    """
    delta_t = guess_delta_t(from_phase, to_phase, potential, T)

    normalisation = 0. if ground_state_energy is None else ground_state_energy
    hvt = _make_hydro_vars(to_phase, potential, T, delta_t, normalisation)
    hvf = _make_hydro_vars(from_phase, potential, T, delta_t, normalisation)

    cosmological_energy_density = _cosmological_energy_density(
        hvf, hvt, potential, T, ground_state_energy
    )

    return HydroVars(
        *hvf.values(), *hvt.values(), T, cosmological_energy_density,
        ground_state_energy is None
    )


def cosmological_energy_density(
        from_phase: Phase, to_phase: Phase, potential, T: float,
        ground_state_energy=0.) -> float:
    """
    @returns The false-phase energy density used to calculate the Hubble rate.
    """
    delta_t = guess_delta_t(from_phase, to_phase, potential, T)
    normalisation = 0. if ground_state_energy is None else ground_state_energy
    hvf = _make_hydro_vars(
        from_phase, potential, T, delta_t, normalisation
    )

    if ground_state_energy is not None:
        return hvf["e"]

    hvt = _make_hydro_vars(to_phase, potential, T, delta_t, normalisation)
    return _cosmological_energy_density(
        hvf, hvt, potential, T, ground_state_energy
    )


def _energy_density(from_phase: Phase, to_phase: Phase,
                    potential, T: float, use_from_phase, ground_state_energy=0.) -> float:
    """
    To and from phase are required to deduce appropriate step in numerical derivative

    @returns Energy density in from or to phase
    """
    delta_t = guess_delta_t(from_phase, to_phase, potential, T)
    phase = from_phase if use_from_phase else to_phase
    return _make_hydro_vars(phase, potential, T, delta_t, ground_state_energy)["e"]


def energy_density_from_phase(*args, **kwargs) -> float:
    return _energy_density(*args, **kwargs, use_from_phase=True)


def energy_density_to_phase(*args, **kwargs) -> float:
    return _energy_density(*args, **kwargs, use_from_phase=False)
