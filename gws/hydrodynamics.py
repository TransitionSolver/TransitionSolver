"""
Hydrodynamic quantities for phase transitions and gravitational waves
=====================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from numpy import pi

from analysis.phase_structure import Phase
from models.analysable_potential import AnalysablePotential
from .numdiff import derivatives


GRAV_CONST = 6.7088e-39


@dataclass
class HydroVars:
    """
    Represent hydrodynnmic variables in true and false vacuum
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
    def K(self):
        """
        @returns Kinetic energy fraction, assuming kappa = 1
        """
        return (self.pseudotraceFalse - self.pseudotraceTrue) / \
            self.energyDensityFalse

    @property
    def hubble_constant(self):
        return (8 * pi * GRAV_CONST / 3 * self.energyDensityFalse)**0.5

    def average_pressure_density(self, pf):
        """
        """
        pt = 1 - pf
        return self.pressureFalse * pf + self.pressureTrue * pt

    def adiabatic_index(self, pf):
        """
        Slightly better than averaging the enthalpy of each phase. Use energy conservation for the energy, and average
        the pressure of each phase. Don't use totalEnergyDensity because we should not subtract off the ground state
        energy density
        """
        return 1. + self.average_pressure_density(pf) / self.energyDensityFalse


def interpolate_hydro_vars(
        hv1: HydroVars, hv2: HydroVars, T: float) -> HydroVars:
    """
    @returns Hydrodynamic variables from linear interpolation between two temperatures
    """
    def linear_interpolate(val1: float, val2: float) -> float:
        f = (T - hv1.T) / (hv2.T - hv1.T)
        return val1 + f * (val2 - val1)

    data = [
        linear_interpolate(
            getattr(
                hv1, f.name), getattr(
                hv2, f.name)) for f in fields(HydroVars)]
    return HydroVars(*data)


# TODO: this probably belongs elsewhere, maybe in PhaseStructure or
# AnalysablePotential.
def getTstep(from_phase: Phase, to_phase: Phase,
             potential: AnalysablePotential, T: float) -> float:
    Tmin = max(from_phase.T[0], to_phase.T[0])
    Tmax = min(from_phase.T[-1], to_phase.T[-1])

    # Make sure the step in either direction doesn't take us past Tmin or where one phase disappears. We don't care
    # about Tc because we can sample in the region Tc < T < Tmax for the
    # purpose of differentiation.
    return min(max(0.0005 * Tmax, 0.0001 * potential.temperatureScale),
               0.5 * (T - Tmin), 0.5 * (Tmax - T))


def _make_hydro_vars(phase, potential, ground_state_energy, T, Tstep):

    def free_energy(x: float) -> float:
        phi = phase.findPhaseAtT(x, potential)
        return potential.freeEnergyDensity(phi, x) - ground_state_energy

    f, df, d2f = derivatives(free_energy, T, Tstep)

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

    return p, e, w, s, c2


def make_hydro_vars(from_phase: Phase, to_phase: Phase, potential: AnalysablePotential, T: float,
                    ground_state_energy=0.) -> HydroVars:
    """
    @returns Hydrodynamical variables in true and false vacuum
    """
    # TODO: if Tstep is too small, dF/dT and d2F/dT2 are zero and the sound
    # speed calculation leads to division by zero.
    Tstep = getTstep(from_phase, to_phase, potential, T)

    hvt = _make_hydro_vars(to_phase, potential, ground_state_energy, T, Tstep)
    hvf = _make_hydro_vars(from_phase, potential, ground_state_energy, T, Tstep)

    return HydroVars(*hvf, *hvt, T)


def _energy_density(from_phase: Phase, to_phase: Phase,
                    potential: AnalysablePotential, T: float, use_from_phase) -> float:
    """
    To and from phase are required to deduce appropriate step in numerical derivative

    @returns Energy density in from or to phase
    """
    Tstep = getTstep(from_phase, to_phase, potential, T)
    phase = from_phase if use_from_phase else to_phase

    def free_energy(x: float) -> float:
        phi = phase.findPhaseAtT(x, potential)
        return potential.freeEnergyDensity(phi, x)

    F, dFdT, _ = derivatives(free_energy, T, Tstep)
    return F - T * dFdT


def energy_density_from_phase(*args, **kwargs) -> float:
    return _energy_density(*args, **kwargs, use_from_phase=True)


def energy_density_to_phase(*args, **kwargs) -> float:
    return _energy_density(*args, **kwargs, use_from_phase=False)
