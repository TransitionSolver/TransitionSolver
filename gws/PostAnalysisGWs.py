import traceback
import json
import numpy as np
from typing import Callable, Type

from analysis.PhaseStructure import Phase
from models.ToyModel import ToyModel
from models.RealScalarSingletModel import RealScalarSingletModel
from models.AnalysablePotential import AnalysablePotential
from analysis import PhaseStructure
import GieseKappa

GRAV_CONST = 6.7088e-39


# Find all transitions that are part of valid transition paths.
def extractRelevantTransitions(report: dict) -> list[dict]:
    relevantTransitions = []

    try:
        isTransitionRelevant = [False] * len(report['transitions'])

        for transitionSequence in report['paths']:
            for transitionID in transitionSequence['transitions']:
                isTransitionRelevant[transitionID] = True

        for transition in report['transitions']:
            if isTransitionRelevant[transition['id']]:
                relevantTransitions.append(transition)

        return relevantTransitions

    except Exception:
        traceback.print_exc()
        return []


def determineGWs(phaseStructure: PhaseStructure.PhaseStructure, transitionReport: dict, potential: AnalysablePotential,
        groundStateEnergyDensity: float) -> tuple[float, float]:
    fromPhase: Phase = phaseStructure.phases[transitionReport['falsePhase']]
    toPhase: Phase = phaseStructure.phases[transitionReport['truePhase']]

    # General form: Omega = redshift * H tau_sw * H tau_c * spectralShape

    T = determineTransitionTemperature(transitionReport)
    vw = determineBubbleWallVelocity()
    K = determineKineticEnergyFraction(fromPhase, toPhase, potential, T, vw)
    if K == 0:
        return 0., 0.
    lenScale = determineLengthScale(transitionReport)

    rhof, rhot = calculateEnergyDensityAtT(fromPhase, toPhase, potential, T)
    wf, wt = calculateEnthalpyDensityAtT(fromPhase, toPhase, potential, T)
    H = np.sqrt(8*np.pi*GRAV_CONST/3*(rhof - groundStateEnergyDensity))

    # Weight the enthalpy by the fraction of the Universe in each phase. This will underestimate the enthalpy because
    # we neglect the reheated regions around the bubble wall.
    # TODO: maybe we can get this from Giese's code?
    averageEnthalpy = wf*0.71 + 0.29*wt

    # Assuming energy conservation so averageEnergy = rhof.
    adiabaticIndex = averageEnthalpy / rhof

    fluidVelocity = np.sqrt(K/adiabaticIndex)
    # Assume the rotational modes are negligible.
    fluidVelocityLong = fluidVelocity

    tau_sw = lenScale / fluidVelocityLong
    upsilon = 1 - 1 / np.sqrt(1 + 2*H*tau_sw)

    csfSq, cstSq = calculateSoundSpeedSq(fromPhase, toPhase, potential, T)
    soundSpeed = np.sqrt(csfSq)
    tau_c = lenScale / soundSpeed

    redshift = 1.67e-5 * (100/potential.ndof)**(1./3.)

    # General form:
    Omega_peak = redshift * K*K * upsilon * H*tau_c
    print('Omega peak (general):', Omega_peak)
    # Fit from our GW review (but dividing length scale by soundSpeed in accordance with updated estimate of tau_c).
    #Omega_peak = 2.59e-6*(100/potential.ndof)**(1./3.) * K*K * H*(lenScale/(8*np.pi)**(1./3.))/soundSpeed * upsilon
    Omega_peak = 0.15509*redshift * K*K * H*(lenScale/(8*np.pi)**(1./3.))/soundSpeed * upsilon
    print('Omega peak (fit):', Omega_peak)
    zp = 10.  # This assumes the peak frequency corresponds to 10*lenScale. This result comes from simulations
    #   (https://arxiv.org/pdf/1704.05871.pdf) and is expected to change if vw ~ vCJ (specifically zp will increase).
    f_peak = 8.9e-6*(potential.ndof/100)**(1./6.)*(T/100)/(H*lenScale)*(zp/10)

    return Omega_peak, f_peak


def spectralShape(f: float, f_peak: float) -> float:
    x = f/f_peak
    return x**3 * (7 / (4 + 3*x**2))**3.5


def getGWfunc(phaseStructure: PhaseStructure.PhaseStructure, transitionReport: dict, potential: AnalysablePotential,
        groundStateEnergyDensity: float) -> Callable[[float], float]:
    Omega_peak, f_peak = determineGWs(phaseStructure, transitionReport, potential, groundStateEnergyDensity)
    return lambda f: Omega_peak*spectralShape(f, f_peak)


def determineTransitionTemperature(transition: dict) -> float:
    return transition['Tp']


# TODO: Just pick a value for now. Should really read from the transition report but we can't trust that result anyway.
#  We can't use vw = 1 because it breaks Giese's code for kappa.
def determineBubbleWallVelocity() -> float:
    return 0.95


def determineKineticEnergyFraction(fromPhase: Phase, toPhase: Phase, potential: AnalysablePotential, T: float,
        vw: float) -> float:
    pf, pt = calculatePressureAtT(fromPhase, toPhase, potential, T)
    rhof, rhot = calculateEnergyDensityAtT(fromPhase, toPhase, potential, T)
    wf, wt = calculateEnthalpyDensityAtT(fromPhase, toPhase, potential, T)
    csfSq, cstSq = calculateSoundSpeedSq(fromPhase, toPhase, potential, T)

    # Pseudo-trace.
    thetaf = (rhof - pf/cstSq) / 4
    thetat = (rhot - pt/cstSq) / 4

    alpha = 4*(thetaf - thetat) / (3*wf)

    kappa = GieseKappa.kappaNuMuModel(cstSq, csfSq, alpha, vw)

    return (thetaf - thetat) / rhof * kappa


def determineLengthScale(transitionReport: dict) -> float:
    return transitionReport['meanBubbleSeparation']


def calculatePressureAtT(fromPhase: Phase, toPhase: Phase, potential: AnalysablePotential, T: float) -> tuple[float,
        float]:
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


def calculateEnergyDensityAtT(fromPhase: Phase, toPhase: Phase, potential: AnalysablePotential, T: float) ->\
        tuple[float, float]:
    Tmin = max(fromPhase.T[0], toPhase.T[0])
    Tmax = min(fromPhase.T[-1], toPhase.T[-1])

    # Make sure the step in either direction doesn't take us past Tmin or where one phase disappears. We don't care
    # about Tc because we can sample in the region Tc < T < Tmax for the purpose of differentiation.
    Tstep = min(max(0.0005*Tmax, 0.0001*potential.temperatureScale), 0.5*(T - Tmin), 0.5*(Tmax - T))

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

    return ef, et


def calculateEnthalpyDensityAtT(fromPhase: Phase, toPhase: Phase, potential: AnalysablePotential, T: float) ->\
        tuple[float, float]:
    Tmin = max(fromPhase.T[0], toPhase.T[0])
    Tmax = min(fromPhase.T[-1], toPhase.T[-1])

    # Make sure the step in either direction doesn't take us past Tmin or where one phase disappears. We don't care
    # about Tc because we can sample in the region Tc < T < Tmax for the purpose of differentiation.
    Tstep = min(max(0.0005*Tmax, 0.0001*potential.temperatureScale), 0.5*(T - Tmin), 0.5*(Tmax - T))

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
    Tmin = max(fromPhase.T[0], toPhase.T[0])
    Tmax = min(fromPhase.T[-1], toPhase.T[-1])

    # Make sure the step in either direction doesn't take us past Tmin or where one phase disappears. We don't care
    # about Tc because we can sample in the region Tc < T < Tmax for the purpose of differentiation.
    Tstep = min(max(0.0005*Tmax, 0.0001*potential.temperatureScale), 0.5*(T - Tmin), 0.5*(Tmax - T))

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


def main(potentialClass: Type[AnalysablePotential], folderName: str) -> None:
    with open(folderName + 'phase_history.json', 'r') as f:
        report = json.load(f)

    relevantTransitions = extractRelevantTransitions(report)

    if len(relevantTransitions) == 0:
        print('No relevant transition detected.')
        return

    bSuccess, phaseStructure = PhaseStructure.load_data(folderName + 'phase_structure.dat')

    if not bSuccess:
        print('Failed to load phase structure.')
        return

    potential = potentialClass(*np.loadtxt(folderName + 'parameter_point.txt'))

    groundStateEnergyDensity = np.inf

    # Find the ground state energy density.
    for phase in phaseStructure.phases:
        if phase.T[0] == 0 and phase.V[0] < groundStateEnergyDensity:
            groundStateEnergyDensity = phase.V[0]

    for transitionReport in relevantTransitions:
        Omega_peak, f_peak = determineGWs(phaseStructure, transitionReport, potential, groundStateEnergyDensity)
        #gwFunc = getGWfunc(phaseStructure, transitionReport, potential, groundStateEnergyDensity)
        print('Transition ID:', transitionReport['id'])
        print('Peak amplitude:', Omega_peak)
        print('Peak frequency:', f_peak)


if __name__ == "__main__":
    main(RealScalarSingletModel, 'output/RSS/RSS_BP1/')
