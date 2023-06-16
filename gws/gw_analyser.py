import traceback
import json

import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Type, Optional

import scipy.integrate

from analysis.phase_structure import Phase, PhaseStructure
from gws.hydrodynamics import HydroVars
from models.real_scalar_singlet_model import RealScalarSingletModel
from models.real_scalar_singlet_model_ht import RealScalarSingletModel_HT
from models.toy_model import ToyModel
from models.analysable_potential import AnalysablePotential
from analysis import phase_structure
from gws import giese_kappa, hydrodynamics
from gws.detectors.lisa import LISA
from gws.detectors.gw_detector import GWDetector

GRAV_CONST = 6.7088e-39


class GWAnalyser_InidividualTransition:
    potential: AnalysablePotential
    phaseStructure: PhaseStructure
    fromPhase: Phase
    toPhase: Phase
    transitionReport: dict
    hydroVars: HydroVars
    detector: GWDetector
    peakAmplitude: float
    peakFrequency_primary: float
    peakFrequency_secondary: float
    SNR: float

    def __init__(self, phaseStructure: PhaseStructure, transitionReport: dict, potential: AnalysablePotential, detector:
            GWDetector):
        self.transitionReport = transitionReport
        self.phaseStructure = phaseStructure
        self.fromPhase = self.phaseStructure.phases[self.transitionReport['falsePhase']]
        self.toPhase = self.phaseStructure.phases[self.transitionReport['truePhase']]
        self.potential = potential
        self.detector = detector
        self.determineGWs()

    def determineGWs(self) -> tuple[float, float]:
        # General form: Omega = redshift * H tau_sw * H tau_c * spectralShape

        T = self.determineTransitionTemperature()
        self.hydroVars = hydrodynamics.getHydroVars(self.fromPhase, self.toPhase, self.potential, T)
        Treh = self.determineReheatingTemperature()
        vw = self.determineBubbleWallVelocity()
        K = self.determineKineticEnergyFraction(vw)
        if K == 0:
            return 0., 0.
        lenScale_primary = self.determineLengthScale()

        totalEnergyDensity = self.hydroVars.energyDensityFalse - self.phaseStructure.groundStateEnergyDensity

        H = np.sqrt(8*np.pi*GRAV_CONST/3 * totalEnergyDensity)

        # Weight the enthalpy by the fraction of the Universe in each phase. This will underestimate the enthalpy
        # because we neglect the reheated regions around the bubble wall.
        # TODO: maybe we can get this from Giese's code?
        #averageEnthalpyDensity = self.hydroVars.enthalpyDensityFalse*0.71 + 0.29*self.hydroVars.enthalpyDensityTrue
        # Slightly better than averaging the enthalpy of each phase. Use energy conservation for the energy, and average
        # the pressure of each phase. Don't use totalEnergyDensity because we should not subtract off the ground state
        # energy density. The enthalpy is -T*dV/dT, so the ground state energy density doesn't contribute.
        # TODO: fix hard-coded vacuum fractions.
        averageEnthalpyDensity = self.hydroVars.pressureFalse*0.71 + self.hydroVars.pressureTrue*0.29\
            + self.hydroVars.energyDensityFalse

        # Assuming energy conservation so averageEnergyDensity = totalEnergyDensity.
        adiabaticIndex = averageEnthalpyDensity / totalEnergyDensity

        fluidVelocity = np.sqrt(K/adiabaticIndex)
        # Assume the rotational modes are negligible.
        fluidVelocityLong = fluidVelocity

        tau_sw = lenScale_primary / fluidVelocityLong
        upsilon = 1 - 1 / np.sqrt(1 + 2*H*tau_sw)

        soundSpeed = np.sqrt(self.hydroVars.soundSpeedSqFalse)
        #tau_c = lenScale_primary / soundSpeed
        lenScale_secondary = lenScale_primary * abs(vw - np.sqrt(self.hydroVars.soundSpeedSqFalse)) / vw

        ndof = self.potential.getDegreesOfFreedom(self.fromPhase.findPhaseAtT(T, self.potential), T)
        redshift = 1.67e-5 * (100/ndof)**(1./3.)

        # General form:
        #Omega_peak = redshift * K*K * upsilon * H*tau_c
        #print('Omega peak (general):', Omega_peak)

        # Fit from our GW review (but dividing length scale by soundSpeed in accordance with updated estimate of tau_c).
        #Omega_peak = 2.59e-6*(100/potential.ndof)**(1./3.) * K*K * H*(lenScale/(8*np.pi)**(1./3.))/soundSpeed * upsilon
        self.peakAmplitude = 0.15509*redshift * K*K * H*(lenScale_primary/(8*np.pi)**(1./3.))/soundSpeed * upsilon
        zp = 10.  # This assumes the peak frequency corresponds to 10*lenScale. This result comes from simulations
        # (https://arxiv.org/pdf/1704.05871.pdf) and is expected to change if vw ~ vCJ (specifically zp will increase).
        self.peakFrequency_primary = 8.9e-6*(ndof/100)**(1./6.)*(Treh/100)\
            /(H*lenScale_primary/(8*np.pi)**(1./3.))*(zp/10)
        self.peakFrequency_secondary = 8.9e-6*(ndof/100)**(1./6.)*(Treh/100)\
            /(H*lenScale_secondary/(8*np.pi)**(1./3.))*(zp/10)

    def calculateSNR(self, gwFunc: Callable[[float], float]) -> float:
        frequencies = self.detector.sensitivityCurve[0]
        sensitivityCurve = self.detector.sensitivityCurve[1]

        # TODO: vectorise.
        gwAmpl = np.array([gwFunc(f) for f in frequencies])

        integrand = [(gwAmpl[i]/sensitivityCurve[i])**2 for i in range(len(frequencies))]

        integral = scipy.integrate.simpson(integrand, frequencies)

        self.SNR = np.sqrt(self.detector.detectionTime * self.detector.numIndependentChannels * integral)

        return self.SNR

    def spectralShape(self, f: float) -> float:
        x = f/self.peakFrequency_primary
        return x**3 * (7 / (4 + 3*x**2))**3.5

    def spectralShape_doubleBroken(self, f: float):
        # From https://arxiv.org/pdf/2209.13551.pdf (Eq. 2.11), originally from https://arxiv.org/pdf/1909.10040.pdf
        # (Eq. # 5.7).
        b = 1
        rb = self.peakFrequency_primary / self.peakFrequency_secondary
        m = (9*rb**4 + b) / (rb**4 + 1)
        x = f/self.peakFrequency_secondary
        return x**9 * ((1 + rb**4) / (rb**4 + x**4))**((9 - b) / 4) * ((b + 4) / (b + 4 - m + m*x**2))**((b + 4) / 2)

    def getGWfunc(self, doubleBroken: bool = True) -> Callable[[float], float]:
        if doubleBroken:
            return lambda f: self.peakAmplitude*self.spectralShape_doubleBroken(f)
        else:
            return lambda f: self.peakAmplitude*self.spectralShape(f)

    def determineTransitionTemperature(self) -> float:
        return self.transitionReport['Tp']

    def determineReheatingTemperature(self) -> float:
        return self.transitionReport['Treh_p']

    # TODO: Just pick a value for now. Should really read from the transition report but we can't trust that result
    #  anyway. We can't use vw = 1 because it breaks Giese's code for kappa.
    def determineBubbleWallVelocity(self) -> float:
        return 0.95

    def determineKineticEnergyFraction(self, vw: float) -> float:
        # Pseudo-trace.
        thetaf = (self.hydroVars.energyDensityFalse - self.hydroVars.pressureFalse/self.hydroVars.soundSpeedSqTrue) / 4
        thetat = (self.hydroVars.energyDensityTrue - self.hydroVars.pressureTrue/self.hydroVars.soundSpeedSqTrue) / 4

        alpha = 4*(thetaf - thetat) / (3*self.hydroVars.enthalpyDensityFalse)

        kappa = giese_kappa.kappaNuMuModel(self.hydroVars.soundSpeedSqTrue, self.hydroVars.soundSpeedSqFalse, alpha, vw)

        totalEnergyDensity = self.hydroVars.energyDensityFalse - self.phaseStructure.groundStateEnergyDensity

        return (thetaf - thetat) / totalEnergyDensity * kappa

    def determineLengthScale(self) -> float:
        return self.transitionReport['meanBubbleSeparation']


class GWAnalyser:
    peakAmplitude: float
    peakFrequency_primary: float
    peakFrequency_secondary: float
    SNR: float
    detector: Optional[GWDetector]
    potential: AnalysablePotential
    phaseHistoryReport: dict
    relevantTransitions: list[dict]

    def __init__(self, detectorClass: Type[GWDetector], potentialClass: Type[AnalysablePotential], outputFolder: str):
        self.peakAmplitude = 0.
        self.peakFrequency_primary = 0.
        self.peakFrequency_secondary = 0.
        self.SNR = 0.
        self.detector = detectorClass()

        with open(outputFolder + 'phase_history.json', 'r') as f:
            self.phaseHistoryReport = json.load(f)

        self.relevantTransitions = extractRelevantTransitions(self.phaseHistoryReport)

        if len(self.relevantTransitions) == 0:
            print('No relevant transition detected.')
            return

        bSuccess, self.phaseStructure = phase_structure.load_data(outputFolder + 'phase_structure.dat')

        if not bSuccess:
            print('Failed to load phase structure.')
            return

        self.potential = potentialClass(*np.loadtxt(outputFolder + 'parameter_point.txt'))

        if len(self.detector.sensitivityCurve) == 0:
            frequencies = np.logspace(-8, 3, 1000)
            self.detector.constructSensitivityCurve(frequencies)

        for transitionReport in self.relevantTransitions:
            gws = GWAnalyser_InidividualTransition(self.phaseStructure, transitionReport, self.potential,
                self.detector)
            gwFunc_single = gws.getGWfunc(doubleBroken=False)
            gwFunc_double = gws.getGWfunc(doubleBroken=True)
            SNR_single = gws.calculateSNR(gwFunc_single)
            SNR_double = gws.calculateSNR(gwFunc_double)
            print('Transition ID:', transitionReport['id'])
            print('Peak amplitude:', gws.peakAmplitude)
            print('Peak frequency (primary):', gws.peakFrequency_primary)
            print('Peak frequency (secondary):', gws.peakFrequency_secondary)
            print('SNR (single):', SNR_single)
            print('SNR (double):', SNR_double)

            plt.loglog(self.detector.sensitivityCurve[0], self.detector.sensitivityCurve[1])
            plt.loglog(self.detector.sensitivityCurve[0], np.array([gwFunc_single(f) for f in
                self.detector.sensitivityCurve[0]]))
            plt.loglog(self.detector.sensitivityCurve[0], np.array([gwFunc_double(f) for f in
                self.detector.sensitivityCurve[0]]))
            plt.legend(['noise', 'single', 'double'])
            plt.margins(0, 0)
            plt.show()


def hydroTester(potentialClass: Type[AnalysablePotential], outputFolder: str):
    with open(outputFolder + 'phase_history.json', 'r') as f:
        phaseHistoryReport = json.load(f)

    relevantTransitions = extractRelevantTransitions(phaseHistoryReport)

    if len(relevantTransitions) == 0:
        print('No relevant transition detected.')
        return

    bSuccess, phaseStructure = phase_structure.load_data(outputFolder + 'phase_structure.dat')

    if not bSuccess:
        print('Failed to load phase structure.')
        return

    potential = potentialClass(*np.loadtxt(outputFolder + 'parameter_point.txt'))

    for transitionReport in relevantTransitions:
        fromPhase = phaseStructure.phases[transitionReport['falsePhase']]
        toPhase = phaseStructure.phases[transitionReport['truePhase']]
        Tc = phaseStructure.transitions[transitionReport['id']].Tc

        pf = []
        pt = []
        ef = []
        et = []
        wf = []
        wt = []
        csfSq = []
        cstSq = []
        gf = []
        gt = []
        T = np.linspace(0, Tc*0.99, 100)

        for t in T:
            hydroVars = hydrodynamics.getHydroVars(fromPhase, toPhase, potential, t)
            pf.append(hydroVars.pressureFalse + phaseStructure.groundStateEnergyDensity)
            pt.append(hydroVars.pressureTrue + phaseStructure.groundStateEnergyDensity)
            ef.append(hydroVars.energyDensityFalse - phaseStructure.groundStateEnergyDensity)
            et.append(hydroVars.energyDensityTrue - phaseStructure.groundStateEnergyDensity)
            wf.append(hydroVars.enthalpyDensityFalse)
            wt.append(hydroVars.enthalpyDensityTrue)
            csfSq.append(hydroVars.soundSpeedSqFalse)
            cstSq.append(hydroVars.soundSpeedSqTrue)
            gf.append(wf[-1] / ef[-1])
            gt.append(wt[-1] / et[-1])

        plt.plot(T, pf)
        plt.plot(T, pt)
        #plt.axhline(-phaseStructure.groundStateEnergyDensity, ls='--')
        plt.xlabel('$T$')
        plt.ylabel('$p(T)$')
        plt.legend(['False', 'True'])
        plt.margins(0, 0)
        plt.show()

        plt.plot(T, ef)
        plt.plot(T, et)
        #plt.axhline(phaseStructure.groundStateEnergyDensity, ls='--')
        plt.xlabel('$T$')
        plt.ylabel('$\\rho(T) - \\rho_{\\mathrm{gs}}$')
        plt.legend(['False', 'True'])
        plt.margins(0, 0)
        plt.show()

        plt.plot(T, wf)
        plt.plot(T, wt)
        plt.xlabel('$T$')
        plt.ylabel('$w(T)$')
        plt.legend(['False', 'True'])
        plt.margins(0, 0)
        plt.show()

        plt.plot(T, csfSq)
        plt.plot(T, cstSq)
        plt.xlabel('$T$')
        plt.ylabel('$c_s(T)$')
        plt.legend(['False', 'True'])
        plt.margins(0, 0)
        plt.show()

        plt.plot(T, gf)
        plt.plot(T, gt)
        plt.xlabel('$T$')
        plt.ylabel('$\\Gamma(T)$')
        plt.legend(['False', 'True'])
        plt.margins(0, 0)
        plt.show()


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


def main(detectorClass, potentialClass, outputFolder):
    GWAnalyser(detectorClass, potentialClass, outputFolder)
    #hydroTester(potentialClass, outputFolder)


if __name__ == "__main__":
    main(LISA, RealScalarSingletModel, 'output/RSS/RSS_BP1/')
    #main(LISA, RealScalarSingletModel_HT, 'output/RSS_HT/RSS_HT_BP1/')
    #main(LISA, ToyModel, 'output/Toy/Toy_BP1/')
