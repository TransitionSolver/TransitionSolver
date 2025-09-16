"""
Hydrodynamic quantities for phase transitions and gravitational waves
=====================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable

from analysis.phase_structure import Phase
from models.analysable_potential import AnalysablePotential


@dataclass
class HydroVars:
    """
    Represent hydrodynnmic variables in true and false vacuum
    """
    pressureFalse: float
    pressureTrue: float
    energyDensityFalse: float
    energyDensityTrue: float
    enthalpyDensityFalse: float
    enthalpyDensityTrue: float
    entropyDensityFalse: float
    entropyDensityTrue: float
    soundSpeedSqFalse: float
    soundSpeedSqTrue: float

    @property
    def traceAnomalyFalse(self):
        return (self.energyDensityFalse - 3*self.pressureFalse) / 4

    @property
    def traceAnomalyTrue(self):
        return (self.energyDensityTrue - 3*self.pressureTrue) / 4

    @property
    def pseudotraceFalse(self):
        return (self.energyDensityFalse - self.pressureFalse / self.soundSpeedSqTrue) / 4

    @property
    def pseudotraceTrue(self):
        return (self.energyDensityTrue - self.pressureTrue / self.soundSpeedSqTrue) / 4

    @property
    def alpha(self):
        return 4 / 3 * (self.pseudotraceFalse - self.pseudotraceTrue) / self.enthalpyDensityFalse

    @property
    def soundSpeedTrue(self):
        return self.soundSpeedSqTrue**0.5

    @property
    def cj_velocity(self):
        return (1. + np.sqrt(3. * self.alpha * (1. - self.soundSpeedSqTrue + 3. * self.soundSpeedSqTrue * self.alpha))) / (1. / self.soundSpeedTrue + 3. * self.soundSpeedTrue * self.alpha)


def interpolate_hydro_vars(hv1: HydroVars, hv2: HydroVars, T1: float, T2: float, T: float) -> HydroVars:
    """
    @returns Hydrodynamic variables from linear interpolation between two temperatures
    """
    def linear_interpolate(val1: float, val2: float) -> float:
        f = (T - T1) / (T2 - T1)
        return val1 + f * (val2 - val1)

    data = [linear_interpolate(getattr(hv1, f.name), getattr(hv2, f.name)) for f in fields(HydroVars)]
    return HydroVars(*data)


# TODO: this probably belongs elsewhere, maybe in PhaseStructure or AnalysablePotential.
def getTstep(fromPhase: Phase, toPhase: Phase, potential: AnalysablePotential, T: float) -> float:
    Tmin = max(fromPhase.T[0], toPhase.T[0])
    Tmax = min(fromPhase.T[-1], toPhase.T[-1])

    # Make sure the step in either direction doesn't take us past Tmin or where one phase disappears. We don't care
    # about Tc because we can sample in the region Tc < T < Tmax for the purpose of differentiation.
    return min(max(0.0005*Tmax, 0.0001*potential.temperatureScale), 0.5*(T - Tmin), 0.5*(Tmax - T))


def dfdx(f: Callable[[float], float], x: float, dx: float, order=4) -> float:
    if order < 4:
        return (f(x+dx) - f(x-dx)) / (2*dx)
    else:
        return (-f(x+2*dx) + 8*f(x+dx) - 8*f(x-dx) + f(x-2*dx)) / (12*dx)


def d2fdx2(f: Callable[[float], float], x: float, dx: float, order=4) -> float:
    if order < 4:
        return (f(x+dx) - 2*f(x) + f(x-dx)) / (dx*dx)
    else:
        return (-f(x+2*dx) + 16*f(x+dx) - 30*f(x) + 16*f(x-dx) - f(x-2*dx)) / (12*dx*dx)


# Calculates all hydrodynamic variables using three samples of the potential. If more than one hydrodynamic variable is
# required, this function is more efficient than using the function for individual hydrodynamic variables separately.
#def getHydroVars(fromPhase: Phase, toPhase: Phase, potential: AnalysablePotential, T: float, order=4) -> HydroVars:
#    if order <= 4:
#        return getHydroVars_secondOrder(fromPhase, toPhase, potential, T)
#    else:
#        return getHydroVars_fourthOrder(fromPhase, toPhase, potential, T)


def getHydroVars_new(fromPhase: Phase, toPhase: Phase, potential: AnalysablePotential, T: float,
        groundStateEnergyDensity: float, order=4) -> HydroVars:
    # TODO: if Tstep is too small, dF/dT and d2F/dT2 are zero and the sound speed calculation leads to division by zero.
    Tstep = getTstep(fromPhase, toPhase, potential, T)

    def FED_f(x: float) -> float:
        phi = fromPhase.findPhaseAtT(x, potential)
        return potential.freeEnergyDensity(phi, x) - groundStateEnergyDensity

    def FED_t(x: float) -> float:
        phi = toPhase.findPhaseAtT(x, potential)
        return potential.freeEnergyDensity(phi, x) - groundStateEnergyDensity

    # Field configuration for the two phases at 3 temperatures.
    """phifl = fromPhase.findPhaseAtT(Tl, potential)
    phifm = fromPhase.findPhaseAtT(T, potential)
    phifh = fromPhase.findPhaseAtT(Th, potential)
    phitl = toPhase.findPhaseAtT(Tl, potential)
    phitm = toPhase.findPhaseAtT(T, potential)
    phith = toPhase.findPhaseAtT(Th, potential)

    # Free energy density of the two phases at those 3 temperatures.
    Ffl = potential.freeEnergyDensity(phifl, Tl)
    Ffm = potential.freeEnergyDensity(phifm, T)
    Ffh = potential.freeEnergyDensity(phifh, Th)
    Ftl = potential.freeEnergyDensity(phitl, Tl)
    Ftm = potential.freeEnergyDensity(phitm, T)
    Fth = potential.freeEnergyDensity(phith, Th)"""

    dFfdT = dfdx(FED_f, T, Tstep, order)
    dFtdT = dfdx(FED_t, T, Tstep, order)

    d2FfdT2 = d2fdx2(FED_f, T, Tstep, order)
    d2FtdT2 = d2fdx2(FED_t, T, Tstep, order)

    # Central difference method for the temperature derivative of the free energy density.
    """dFfdT = (Ffh - Ffl) / (2*Tstep)
    dFtdT = (Fth - Ftl) / (2*Tstep)

    # Central difference method for the second temperature derivative of the free energy density.
    d2FfdT2 = (Ffh - 2*Ffm + Ffl) / Tstep**2
    d2FtdT2 = (Fth - 2*Ftm + Ftl) / Tstep**2"""

    Ffm = FED_f(T)
    Ftm = FED_t(T)

    # Pressure.
    pf = -Ffm
    pt = -Ftm

    # Energy density.
    ef = Ffm - T*dFfdT
    et = Ftm - T*dFtdT

    # Enthalpy density.
    wf = -T*dFfdT
    wt = -T*dFtdT

    # Entropy density.
    sf = -dFfdT
    st = -dFtdT

    # Sound speed squared.
    csfSq = dFfdT / (T*d2FfdT2)
    cstSq = dFtdT / (T*d2FtdT2)

    return HydroVars(pf, pt, ef, et, wf, wt, sf, st, csfSq, cstSq)


def getHydroVars(fromPhase: Phase, toPhase: Phase, potential: AnalysablePotential, T: float) -> HydroVars:
    Tstep = getTstep(fromPhase, toPhase, potential, T)
    Tl = T - Tstep
    Th = T + Tstep

    # Field configuration for the two phases at 3 temperatures.
    phifl = fromPhase.findPhaseAtT(Tl, potential)
    phifm = fromPhase.findPhaseAtT(T, potential)
    phifh = fromPhase.findPhaseAtT(Th, potential)
    phitl = toPhase.findPhaseAtT(Tl, potential)
    phitm = toPhase.findPhaseAtT(T, potential)
    phith = toPhase.findPhaseAtT(Th, potential)

    # Free energy density of the two phases at those 3 temperatures.
    Ffl = potential.freeEnergyDensity(phifl, Tl)
    Ffm = potential.freeEnergyDensity(phifm, T)
    Ffh = potential.freeEnergyDensity(phifh, Th)
    Ftl = potential.freeEnergyDensity(phitl, Tl)
    Ftm = potential.freeEnergyDensity(phitm, T)
    Fth = potential.freeEnergyDensity(phith, Th)

    # Central difference method for the temperature derivative of the free energy density.
    dFfdT = (Ffh - Ffl) / (2*Tstep)
    dFtdT = (Fth - Ftl) / (2*Tstep)

    # Central difference method for the second temperature derivative of the free energy density.
    d2FfdT2 = (Ffh - 2*Ffm + Ffl) / Tstep**2
    d2FtdT2 = (Fth - 2*Ftm + Ftl) / Tstep**2

    # Pressure.
    pf = -Ffm
    pt = -Ftm

    # Energy density.
    ef = Ffm - T*dFfdT
    et = Ftm - T*dFtdT

    # Enthalpy density.
    wf = -T*dFfdT
    wt = -T*dFtdT

    # Entropy density.
    sf = -dFfdT
    st = -dFtdT

    # Sound speed squared.
    csfSq = dFfdT / (T*d2FfdT2)
    cstSq = dFtdT / (T*d2FtdT2)

    return HydroVars(pf, pt, ef, et, wf, wt, sf, st, csfSq, cstSq)


def calculatePressureAtT(fromPhase: Phase, toPhase: Phase, potential: AnalysablePotential, T: float)\
        -> tuple[float, float]:
    # Field configuration for the two phases.
    phif = fromPhase.findPhaseAtT(T, potential)
    phit = toPhase.findPhaseAtT(T, potential)

    # Free energy density of the two phases.
    Ff = potential.freeEnergyDensity(phif, T)
    Ft = potential.freeEnergyDensity(phit, T)

    # Pressure.
    pf = -Ff
    pt = -Ft

    return pf, pt


"""def calculateEnergyDensityAtT_old(fromPhase: Phase, toPhase: Phase, potential: AnalysablePotential, T: float) ->\
        tuple[float, float]:
    Tstep = getTstep(fromPhase, toPhase, potential, T)

    Tl = T - Tstep
    Th = T + Tstep

    # Field configuration for the two phases at 3 temperatures.
    phifl = fromPhase.findPhaseAtT(Tl, potential)
    phifm = fromPhase.findPhaseAtT(T, potential)
    phifh = fromPhase.findPhaseAtT(Th, potential)
    phitl = toPhase.findPhaseAtT(Tl, potential)
    phitm = toPhase.findPhaseAtT(T, potential)
    phith = toPhase.findPhaseAtT(Th, potential)

    # Free energy density of the two phases at those 3 temperatures.
    Ffl = potential.freeEnergyDensity(phifl, Tl)
    Ffm = potential.freeEnergyDensity(phifm, T)
    Ffh = potential.freeEnergyDensity(phifh, Th)
    Ftl = potential.freeEnergyDensity(phitl, Tl)
    Ftm = potential.freeEnergyDensity(phitm, T)
    Fth = potential.freeEnergyDensity(phith, Th)

    # Central difference method for the temperature derivative of the free energy density.
    dFfdT = (Ffh - Ffl) / (2*Tstep)
    dFtdT = (Fth - Ftl) / (2*Tstep)

    # Energy density.
    ef = Ffm - T*dFfdT
    et = Ftm - T*dFtdT

    return ef, et"""


def calculateEnergyDensityAtT(fromPhase: Phase, toPhase: Phase, potential: AnalysablePotential, T: float) ->\
        tuple[float, float]:
    Tstep = getTstep(fromPhase, toPhase, potential, T)

    ef = calculateEnergyDensityAtT_singlePhase_supplied(fromPhase, potential, T, Tstep)
    et = calculateEnergyDensityAtT_singlePhase_supplied(toPhase, potential, T, Tstep)

    return ef, et


def calculateEnergyDensityAtT_singlePhase(fromPhase: Phase, toPhase: Phase, potential: AnalysablePotential, T: float,
        forFromPhase: bool = True) -> float:
    Tstep = getTstep(fromPhase, toPhase, potential, T)
    phase = fromPhase if forFromPhase else toPhase

    return calculateEnergyDensityAtT_singlePhase_supplied(phase, potential, T, Tstep)


def calculateEnergyDensityAtT_singlePhase_supplied(phase: Phase, potential: AnalysablePotential, T: float,
        Tstep: float) -> float:
    Tl = T - Tstep
    Th = T + Tstep

    # Field configuration for the phase at 3 temperatures.
    phil = phase.findPhaseAtT(Tl, potential)
    phim = phase.findPhaseAtT(T, potential)
    phih = phase.findPhaseAtT(Th, potential)

    # Free energy density of the phase at those 3 temperatures.
    Fl = potential.freeEnergyDensity(phil, Tl)
    Fm = potential.freeEnergyDensity(phim, T)
    Fh = potential.freeEnergyDensity(phih, Th)

    # Central difference method for the temperature derivative of the free energy density.
    dFdT = (Fh - Fl) / (2*Tstep)

    # Energy density.
    return Fm - T*dFdT


def calculateEnthalpyDensityAtT(fromPhase: Phase, toPhase: Phase, potential: AnalysablePotential, T: float) ->\
        tuple[float, float]:
    Tstep = getTstep(fromPhase, toPhase, potential, T)
    Tl = T - Tstep
    Th = T + Tstep

    # Field configuration for the two phases at 2 temperatures.
    phifl = fromPhase.findPhaseAtT(Tl, potential)
    phifh = fromPhase.findPhaseAtT(Th, potential)
    phitl = toPhase.findPhaseAtT(Tl, potential)
    phith = toPhase.findPhaseAtT(Th, potential)

    # Free energy density of the two phases at those 2 temperatures.
    Ffl = potential.freeEnergyDensity(phifl, Tl)
    Ffh = potential.freeEnergyDensity(phifh, Th)
    Ftl = potential.freeEnergyDensity(phitl, Tl)
    Fth = potential.freeEnergyDensity(phith, Th)

    # Central difference method for the temperature derivative of the free energy density.
    dFfdT = (Ffh - Ffl) / (2*Tstep)
    dFtdT = (Fth - Ftl) / (2*Tstep)

    # Enthalpy density.
    wf = -T*dFfdT
    wt = -T*dFtdT

    return wf, wt


def calculateSoundSpeedSq(fromPhase: Phase, toPhase: Phase, potential: AnalysablePotential, T: float) -> tuple[float,
        float]:
    Tstep = getTstep(fromPhase, toPhase, potential, T)
    Tl = T - Tstep
    Th = T + Tstep

    # Field configuration for the two phases at 3 temperatures.
    phifl = fromPhase.findPhaseAtT(Tl, potential)
    phifm = fromPhase.findPhaseAtT(T, potential)
    phifh = fromPhase.findPhaseAtT(Th, potential)
    phitl = toPhase.findPhaseAtT(Tl, potential)
    phitm = toPhase.findPhaseAtT(T, potential)
    phith = toPhase.findPhaseAtT(Th, potential)

    # Free energy density of the two phases at those 3 temperatures.
    Ffl = potential.freeEnergyDensity(phifl, Tl)
    Ffm = potential.freeEnergyDensity(phifm, T)
    Ffh = potential.freeEnergyDensity(phifh, Th)
    Ftl = potential.freeEnergyDensity(phitl, Tl)
    Ftm = potential.freeEnergyDensity(phitm, T)
    Fth = potential.freeEnergyDensity(phith, Th)

    # Central difference method for the temperature derivative of the free energy density.
    dFfdT = (Ffh - Ffl) / (2*Tstep)
    dFtdT = (Fth - Ftl) / (2*Tstep)

    # Central difference method for the second temperature derivative of the free energy density.
    d2FfdT2 = (Ffh - 2*Ffm + Ffl) / Tstep**2
    d2FtdT2 = (Fth - 2*Ftm + Ftl) / Tstep**2

    # Sound speed squared.
    csfSq = dFfdT / (T*d2FfdT2)
    cstSq = dFtdT / (T*d2FtdT2)

    return csfSq, cstSq
