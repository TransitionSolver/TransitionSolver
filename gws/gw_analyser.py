from __future__ import annotations
import time
import traceback
import json

import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Type, Optional, List

import scipy.integrate
import scipy.optimize

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
from scipy.interpolate import CubicSpline

GRAV_CONST = 6.7088e-39


class GWAnalysisSettings:
    bUseBubbleSeparation: bool
    sampleIndex: int
    suppliedRho_t: Optional[Callable[[float], float]]

    def __init__(self):
        self.bUseBubbleSeparation = True
        self.sampleIndex = -1
        self.suppliedRho_t = None


class GWAnalyser_InidividualTransition:
    potential: AnalysablePotential
    phaseStructure: PhaseStructure
    fromPhase: Phase
    toPhase: Phase
    transitionReport: dict
    hydroVars: HydroVars
    detector: GWDetector
    peakAmplitude: float = 0.
    peakFrequency_primary: float = 0.
    peakFrequency_secondary: float = 0.
    SNR: float = 0.
    K: float = 0.
    vw: float = 0.
    T: float = 0.
    Treh: float = 0.
    lenScale_primary: float = 0.
    lenScale_secondary: float = 0.
    adiabaticIndex: float = 0.
    fluidVelocity: float = 0.
    upsilon: float = 0.
    soundSpeed: float = 0.
    ndof: float = 0.
    redshift: float = 0.

    def __init__(self, phaseStructure: PhaseStructure, transitionReport: dict, potential: AnalysablePotential, detector:
            GWDetector):
        self.transitionReport = transitionReport
        self.phaseStructure = phaseStructure
        self.fromPhase = self.phaseStructure.phases[self.transitionReport['falsePhase']]
        self.toPhase = self.phaseStructure.phases[self.transitionReport['truePhase']]
        self.potential = potential
        self.detector = detector

    # Index is the index of the subsampled thermal parameters to use. If the index is negative or out of bounds, the
    # thermal parameters at the percolation temperature will be used.
    def determineGWs(self, settings: GWAnalysisSettings = None) -> tuple[float, float]:
        # General form: Omega = redshift * H tau_sw * H tau_c * spectralShape

        # If no settings are supplied, use default settings.
        if settings is None:
            settings = GWAnalysisSettings()

        if settings.sampleIndex >= 0 and (not ('TSubsample' in self.transitionReport) or settings.sampleIndex >=
                len(self.transitionReport['TSubsample'])):
            settings.sampleIndex = -1

        self.T = self.determineTransitionTemperature(settings)
        self.hydroVars = hydrodynamics.getHydroVars(self.fromPhase, self.toPhase, self.potential, self.T)
        self.Treh = self.determineReheatingTemperature(settings)
        self.vw = self.determineBubbleWallVelocity()
        self.K = self.determineKineticEnergyFraction()
        if self.K == 0:
            return 0., 0.
        self.lenScale_primary = self.determineLengthScale(settings)

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
        self.adiabaticIndex = averageEnthalpyDensity / totalEnergyDensity

        self.fluidVelocity = np.sqrt(self.K/self.adiabaticIndex)
        # Assume the rotational modes are negligible.
        fluidVelocityLong = self.fluidVelocity

        tau_sw = self.lenScale_primary / fluidVelocityLong
        self.upsilon = 1 - 1 / np.sqrt(1 + 2*H*tau_sw)
        self.upsilon_old = min(1,H*tau_sw)

        self.soundSpeed = np.sqrt(self.hydroVars.soundSpeedSqFalse)
        #tau_c = lenScale_primary / soundSpeed
        self.lenScale_secondary = self.lenScale_primary * abs(self.vw - np.sqrt(self.hydroVars.soundSpeedSqFalse))\
            / self.vw

        self.ndof = self.potential.getDegreesOfFreedom(self.toPhase.findPhaseAtT(self.T, self.potential), self.T)
        self.redshift = 1.67e-5 * (100/self.ndof)**(1./3.)

        # General form:
        #Omega_peak = redshift * K*K * upsilon * H*tau_c
        #print('Omega peak (general):', Omega_peak)

        # Fit from our GW review (but dividing length scale by soundSpeed in accordance with updated estimate of tau_c).
        #Omega_peak = 2.59e-6*(100/potential.ndof)**(1./3.) * K*K * H*(lenScale/(8*np.pi)**(1./3.))/soundSpeed * upsilon
        self.peakAmplitudesw = 0.15509*self.redshift * self.K*self.K * H*(self.lenScale_primary/(8*np.pi)**(1./3.))\
            / self.soundSpeed * self.upsilon
        zp = 10.  # This assumes the peak frequency corresponds to 10*lenScale. This result comes from simulations
        # (https://arxiv.org/pdf/1704.05871.pdf) and is expected to change if vw ~ vCJ (specifically zp will increase).
        self.peakFrequency_primarysw = 8.9e-6 * (self.ndof/100)**(1./6.) * (self.Treh/100)\
            / (H*self.lenScale_primary/(8*np.pi)**(1./3.)) * (zp/10)
        self.peakFrequency_secondarysw = 8.9e-6 * (self.ndof/100)**(1./6.) * (self.Treh/100)\
            / (H*self.lenScale_secondary/(8*np.pi)**(1./3.)) * (zp/10)
            
            
            
        #self.peakAmplitudeturb = 3.35 * 10e-4 * (100/self.ndof)**(1./3.)*(0.1*self.K)**(3/2) * H*(self.lenScale_primary/(8*np.pi)**(1./3.))/self.soundSpeed    #计算turbulence
        self.peakAmplitudeturb = 3.35 * 10e-4 * (100/self.ndof)**(1./3.)* (0.05*self.K)**(3/2) * H*(self.lenScale_primary/(8*np.pi)**(1./3.))/self.soundSpeed
        #self.peakAmplitudeturb = 3.35 * 10e-4 * (100/self.ndof)**(1./3.)* ((1 - self.upsilon)**(2/3)*self.K)**(3/2) * H*(self.lenScale_primary/(8*np.pi)**(1./3.))/self.soundSpeed
        #self.peakAmplitudeturb = 3.35 * 10e-4 * (100/self.ndof)**(1./3.)*((1 - self.upsilon_old)**(2/3)*self.K)**(3/2) * H*(self.lenScale_primary/(8*np.pi)**(1./3.))/self.soundSpeed


        self.peakFrequency_primaryturb = 2.7e-5*(self.ndof/100)**(1./6.)*(self.Treh/100)/(H*self.lenScale_primary/(8*np.pi)**(1./3.))
        self.peakFrequency_secondaryturb = 2.7e-5*(self.ndof/100)**(1./6.)*(self.Treh/100)/(H*self.lenScale_secondary/(8*np.pi)**(1./3.))       
        
        

    def calculateSNR(self, gwFunc: Callable[[float], float]) -> float:
        frequencies = self.detector.sensitivityCurve[0]
        sensitivityCurve = self.detector.sensitivityCurve[1]

        # TODO: vectorise.
        gwAmpl = np.array([gwFunc(f) for f in frequencies])

        integrand = [(gwAmpl[i]/sensitivityCurve[i])**2 for i in range(len(frequencies))]

        integral = scipy.integrate.simpson(integrand, frequencies)

        self.SNR = np.sqrt(self.detector.detectionTime * self.detector.numIndependentChannels * integral)

        return self.SNR

    def spectralShapesw(self, f: float) -> float:
        x = f/self.peakFrequency_primarysw
        return x**3 * (7 / (4 + 3*x**2))**3.5

    def spectralShape_doubleBrokensw(self, f: float):
        # From https://arxiv.org/pdf/2209.13551.pdf (Eq. 2.11), originally from https://arxiv.org/pdf/1909.10040.pdf
        # (Eq. # 5.7).
        b = 1
        rb = self.peakFrequency_primarysw / self.peakFrequency_secondarysw
        m = (9*rb**4 + b) / (rb**4 + 1)
        x = f/self.peakFrequency_secondarysw
        return x**9 * ((1 + rb**4) / (rb**4 + x**4))**((9 - b) / 4) * ((b + 4) / (b + 4 - m + m*x**2))**((b + 4) / 2)
          
        
    def spectralShapeturb(self, f: float) -> float:   
        #From https://arxiv.org/abs/1705.01783 Eq.(31)
          #spectralShape 1 turblence
        x = f/self.peakFrequency_primaryturb
        #totalEnergyDensity = self.hydroVars.energyDensityFalse - self.phaseStructure.groundStateEnergyDensity
        #H = np.sqrt(8*np.pi*GRAV_CONST/3 * totalEnergyDensity) #H*
        rf = 16.5*10**(-6)*(self.Treh/100)*(self.ndof/100)**(1./6.)
        #return x**3/((1+x)**(11/3) * (1+8*np.pi*f/H))
        return x**3/((1+x)**(11/3) * (1+8*np.pi*f/rf)) 
        
    def spectralShape_doubleBrokenturb(self, f: float):     #spectralShape 2  turblence
        # From https://arxiv.org/pdf/2209.13551.pdf (Eq. 2.11), originally from https://arxiv.org/pdf/1909.10040.pdf
        # (Eq. # 5.7).
        b = 1
        rb = self.peakFrequency_primaryturb / self.peakFrequency_secondaryturb
        m = (9*rb**4 + b) / (rb**4 + 1)
        x = f/self.peakFrequency_secondaryturb
        return x**9 * ((1 + rb**4) / (rb**4 + x**4))**((9 - b) / 4) * ((b + 4) / (b + 4 - m + m*x**2))**((b + 4) / 2)
              

    def getGWfuncsw(self, doubleBroken: bool = True) -> Callable[[float], float]:
        if self.peakAmplitudesw == 0. or self.peakFrequency_primarysw == 0.:
            return lambda f: 0.

        if doubleBroken:
            return lambda f: self.peakAmplitudesw*self.spectralShape_doubleBrokensw(f)
        else:
            return lambda f: self.peakAmplitudesw*self.spectralShapesw(f)
            
            
    def getGWfuncturb(self) -> Callable[[float], float]:
        if self.peakAmplitudeturb == 0. or self.peakFrequency_primaryturb == 0.:
            return lambda f: 0.
        else:
            return lambda f: self.peakAmplitudeturb*self.spectralShapeturb(f)        
            
            
    def getGWfunctot(self, doubleBroken: bool = True) -> Callable[[float], float]:
    
        if doubleBroken:
            return lambda f: self.peakAmplitudesw*self.spectralShape_doubleBrokensw(f) + self.peakAmplitudeturb*self.spectralShapeturb(f)      
        else:
            return lambda f: self.peakAmplitudesw*self.spectralShapesw(f) + self.peakAmplitudeturb*self.spectralShapeturb(f)
            

    def determineTransitionTemperature(self, settings: GWAnalysisSettings) -> float:
        if settings.sampleIndex < 0:
            return self.transitionReport['Tp']
        else:
            return self.transitionReport['TSubsample'][settings.sampleIndex]

    def determineReheatingTemperature(self, settings: GWAnalysisSettings) -> float:
        if settings.sampleIndex < 0:
            return self.transitionReport['Treh_p']
        else:
            return self.calculateReheatingTemperature(self.transitionReport['TSubsample'][settings.sampleIndex],
                settings.suppliedRho_t)

    # Copied from transition_analysis.TransitionAnalyer.calculateReheatTemperature.
    def calculateReheatingTemperature(self, T: float, suppliedRho_t: Optional[Callable[[float], float]]) -> float:
        Tmin = self.transitionReport['T'][-1]
        Tc = self.transitionReport['Tc']
        Tsep = min(0.001*(Tc - Tmin), 0.5*(T - Tmin))
        rhof = hydrodynamics.calculateEnergyDensityAtT_singlePhase(self.fromPhase, self.toPhase, self.potential, T)
        def objective(t):
            if suppliedRho_t is not None:
                rhot = suppliedRho_t(t)
            else:
                rhot = hydrodynamics.calculateEnergyDensityAtT_singlePhase(self.fromPhase, self.toPhase, self.potential,
                    t, forFromPhase=False)
            # Conservation of energy => rhof = rhof*Pf + rhot*Pt which is equivalent to rhof = rhot (evaluated at
            # different temperatures, T and Tt (Treh), respectively).
            return rhot - rhof

        # If the energy density of the true vacuum is never larger than the current energy density of the false vacuum
        # even at Tc, then reheating goes beyond Tc.
        if objective(Tc) < 0:
            # Also, check the energy density of the true vacuum when it first appears. If a solution still doesn't exist
            # here, then just report -1.
            if self.toPhase.T[-1]-2*Tsep > Tc and objective(self.toPhase.T[-1]-2*Tsep) < 0:
                return -1
            else:
                return scipy.optimize.toms748(objective, T, self.toPhase.T[-1]-2*Tsep)
        return scipy.optimize.toms748(objective, T, Tc)

    # TODO: Just pick a value for now. Should really read from the transition report but we can't trust that result
    #  anyway. We can't use vw = 1 because it breaks Giese's code for kappa.
    def determineBubbleWallVelocity(self) -> float:
        return 0.95

    def determineKineticEnergyFraction(self) -> float:
        if self.hydroVars.soundSpeedSqTrue <= 0:
            return 0.

        # Pseudo-trace.
        thetaf = (self.hydroVars.energyDensityFalse - self.hydroVars.pressureFalse/self.hydroVars.soundSpeedSqTrue) / 4
        thetat = (self.hydroVars.energyDensityTrue - self.hydroVars.pressureTrue/self.hydroVars.soundSpeedSqTrue) / 4

        alpha = 4*(thetaf - thetat) / (3*self.hydroVars.enthalpyDensityFalse)

        kappa = giese_kappa.kappaNuMuModel(self.hydroVars.soundSpeedSqTrue, self.hydroVars.soundSpeedSqFalse, alpha,
            self.vw)

        totalEnergyDensity = self.hydroVars.energyDensityFalse - self.phaseStructure.groundStateEnergyDensity

        return (thetaf - thetat) / totalEnergyDensity * kappa

    def determineLengthScale(self, settings) -> float:
        if settings.sampleIndex < 0:
            if settings.bUseBubbleSeparation:
                return self.transitionReport['meanBubbleSeparation']
            else:
                return self.transitionReport['meanBubbleRadius']
        else:
            if settings.bUseBubbleSeparation:
                return self.transitionReport['meanBubbleSeparationArray'][settings.sampleIndex]
            else:
                return self.transitionReport['meanBubbleRadiusArray'][settings.sampleIndex]


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

    def determineGWs(self):
        for transitionReport in self.relevantTransitions:
            gws = GWAnalyser_InidividualTransition(self.phaseStructure, transitionReport, self.potential,
                self.detector)
            # Determine GWs using default settings.
            gws.determineGWs(settings=None)
            gwFunc_singlesw = gws.getGWfuncsw(doubleBroken=False)
            gwFunc_doublesw = gws.getGWfuncsw(doubleBroken=True)               
            gwFunc_singleturb = gws.getGWfuncturb()
            gwFunc_doubleturb = gws.getGWfuncturb()
            #gwFunc_singletot = gws.getGWfunctot(doubleBroken=False)
            gwFunc_tot = gws.getGWfunctot(doubleBroken=True) 
            
            SNR_singlesw = gws.calculateSNR(gwFunc_singlesw)
            SNR_doublesw = gws.calculateSNR(gwFunc_doublesw)
            SNR_singleturb = gws.calculateSNR(gwFunc_singleturb)
            SNR_doubleturb = gws.calculateSNR(gwFunc_doubleturb)
            SNR_tot = gws.calculateSNR(gwFunc_tot)
            
            print('Transition ID:', transitionReport['id'])
            print('Peak amplitudesw:', gws.peakAmplitudesw)
            print('Peak amplitudeturb:', gws.peakAmplitudeturb)
            print('Peak frequency sw (primary):', gws.peakFrequency_primarysw)
            print('Peak frequency sw (secondary):', gws.peakFrequency_secondarysw)
            print('Peak frequency turb (primary):', gws.peakFrequency_primaryturb)
            print('Peak frequency turb (secondary):', gws.peakFrequency_secondaryturb)
            print('SNR sw (single):', SNR_singlesw)
            print('SNR sw (double):', SNR_doublesw)
            print('SNR turb (single):', SNR_singleturb)
            print('SNR turb (double):', SNR_doubleturb) 
            print('SNR tot (single):', SNR_tot)  
            
            plt.loglog(self.detector.sensitivityCurve[0], self.detector.sensitivityCurve[1])
            #plt.loglog(self.detector.sensitivityCurve[0], np.array([gwFunc_singlesw(f) for f in self.detector.sensitivityCurve[0]]))
            plt.loglog(self.detector.sensitivityCurve[0], np.array([gwFunc_doublesw(f) for f in self.detector.sensitivityCurve[0]]))
            plt.loglog(self.detector.sensitivityCurve[0], np.array([gwFunc_singleturb(f) for f in self.detector.sensitivityCurve[0]]))
            #plt.loglog(self.detector.sensitivityCurve[0], np.array([gwFunc_doubleturb(f) for f in self.detector.sensitivityCurve[0]]))            
            #plt.loglog(self.detector.sensitivityCurve[0], np.array([gwFunc_singletot(f) for f in self.detector.sensitivityCurve[0]])) 
            plt.loglog(self.detector.sensitivityCurve[0], np.array([gwFunc_tot(f) for f in self.detector.sensitivityCurve[0]]))            
            #plt.legend(['noise', 'single sw','double sw', 'single turb' , "tot"])
            plt.legend(['noise', 'double sw', 'single turb' , "tot"])
            plt.margins(0, 0)
            plt.show()
            
            
    def scanGWs(self):
        for transitionReport in self.relevantTransitions:
            startTime = time.perf_counter()

            allT: List[float] = transitionReport['TSubsample']

            # Precompute an interpolated curve for the true vacuum energy density, for use in calculating the reheating
            # temperature. This saves a lot of work. E.g. if the total runtime is 85 seconds, eliminating the Treh
            # calculation altogether takes the runtime down to 15 seconds. Using the supplied rhot for Treh takes the
            # runtime to 20 seconds.
            self.fromPhase = self.phaseStructure.phases[transitionReport['falsePhase']]
            self.toPhase = self.phaseStructure.phases[transitionReport['truePhase']]
            energy_T = np.linspace(allT[-1], transitionReport['Tc'], 100)
            rhot = [hydrodynamics.calculateEnergyDensityAtT_singlePhase(self.fromPhase, self.toPhase, self.potential,
                    t, forFromPhase=False) for t in energy_T]
            rhot_interp = scipy.interpolate.CubicSpline(energy_T, rhot)

            indices = []
            T = []

            print('Num subsamples:', len(allT))

            skipFactor = 1
            for i in range(min(len(allT)//skipFactor-1,len(allT))):
                indices.append(1+i*skipFactor)

            print('Sampling indices:', indices)

            peakAmpsw: List[float] = []
            peakAmpturb: List[float] = []
            peakFreq_primarysw: List[float] = []
            peakFreq_secondarysw: List[float] = []
            peakFreq_primaryturb: List[float] = []
            peakFreq_secondaryturb: List[float] = []
            SNR_singlesw: List[float] = []
            SNR_doublesw: List[float] = []
            SNR_singleturb: List[float] = []
            SNR_doubleturb: List[float] = []
            K: List[float] = []
            vw: List[float] = []
            Treh: List[float] = []
            lenScale_primary: List[float] = []
            lenScale_secondary: List[float] = []
            adiabaticIndex: List[float] = []
            fluidVelocity: List[float] = []
            upsilon: List[float] = []
            csf: List[float] = []
            cst: List[float] = []
            ndof: List[float] = []
            redshift: List[float] = []
            beta: List[float] = []
            lenScale_beta: List[float] = []

            for i in indices:
                gws = GWAnalyser_InidividualTransition(self.phaseStructure, transitionReport, self.potential,
                    self.detector)
                settings = GWAnalysisSettings()
                settings.sampleIndex = i
                settings.suppliedRho_t = rhot_interp
                gws.determineGWs(settings)
                gwFunc_singlesw = gws.getGWfuncsw(doubleBroken=False)
                gwFunc_doublesw = gws.getGWfuncsw(doubleBroken=True)
                gwFunc_singleturb = gws.getGWfuncturb()
                gwFunc_doubleturb = gws.getGWfuncturb()
                
                if gws.peakAmplitudesw > 0:
                    T.append(allT[i])
                    peakAmpsw.append(gws.peakAmplitudesw)
                    peakFreq_primarysw.append(gws.peakFrequency_primarysw)
                    peakFreq_secondarysw.append(gws.peakFrequency_secondarysw)
                    SNR_singlesw.append(gws.calculateSNR(gwFunc_singlesw))
                    SNR_doublesw.append(gws.calculateSNR(gwFunc_doublesw))
                    K.append(gws.K)
                    vw.append(gws.vw)
                    Treh.append(gws.Treh)
                    lenScale_primary.append(gws.lenScale_primary)
                    lenScale_secondary.append(gws.lenScale_secondary)
                    adiabaticIndex.append(gws.adiabaticIndex)
                    fluidVelocity.append(gws.fluidVelocity)
                    upsilon.append(gws.upsilon)
                    csf.append(np.sqrt(gws.hydroVars.soundSpeedSqFalse))
                    cst.append(np.sqrt(gws.hydroVars.soundSpeedSqTrue))
                    ndof.append(gws.ndof)
                    redshift.append(gws.redshift)
                    beta.append(transitionReport['beta'][i])
                    lenScale_beta.append((8*np.pi)**(1/3) * vw[-1] / beta[-1])
                    
                if gws.peakAmplitudeturb > 0:
                    
                    peakAmpturb.append(gws.peakAmplitudeturb)
                    peakFreq_primaryturb.append(gws.peakFrequency_primaryturb)
                    peakFreq_secondaryturb.append(gws.peakFrequency_secondaryturb)
                    SNR_singleturb.append(gws.calculateSNR(gwFunc_singleturb))
                    SNR_doubleturb.append(gws.calculateSNR(gwFunc_doubleturb))
                    

            print('Analysis took:', time.perf_counter() - startTime, 'seconds')

            def plotMilestoneTemperatures():
                if 'Tn' in transitionReport: plt.axvline(transitionReport['Tn'], ls='--', c='r')
                if 'TGammaMax' in transitionReport: plt.axvline(transitionReport['TGammaMax'], ls='--', c='m')
                if 'Tp' in transitionReport: plt.axvline(transitionReport['Tp'], ls='--', c='g')
                if 'Te' in transitionReport: plt.axvline(transitionReport['Te'], ls='--', c='b')
                if 'Tf' in transitionReport: plt.axvline(transitionReport['Tf'], ls='--', c='k')

            def finalisePlot():
                plt.tick_params(size=8, labelsize=18)
                plt.margins(0, 0)
                plt.show()

            plt.rcParams["text.usetex"] = True

            plt.figure(figsize=(12, 8))
            plt.plot(T, SNR_singlesw, lw=2.5)
            plt.plot(T, SNR_doublesw, lw=2.5)
            plt.plot(T, SNR_singleturb, lw=2.5)
            
            plotMilestoneTemperatures()
            plt.xlabel('$T$', fontsize=24)
            plt.ylabel('$\\mathrm{SNR}$', fontsize=24)
            plt.legend(['singlesw', 'doublesw', 'singleturb'], fontsize=20)
            finalisePlot()

            plt.figure(figsize=(12, 8))
            plt.plot(T, peakAmpsw, lw=2.5)
            plotMilestoneTemperatures()
            plt.xlabel('$T$', fontsize=24)
            plt.ylabel('$\\Omega_{\\mathrm{peak_sw}}$', fontsize=24)
            plt.ylim(bottom=0, top=peakAmpsw[-1])
            finalisePlot()
            
            plt.figure(figsize=(12, 8))
            plt.plot(T, peakAmpturb, lw=2.5)
            plotMilestoneTemperatures()
            plt.xlabel('$T$', fontsize=24)
            plt.ylabel('$\\Omega_{\\mathrm{peak_turb}}$', fontsize=24)
            plt.ylim(bottom=0, top=peakAmpsw[-1])
            finalisePlot()

            plt.figure(figsize=(12, 8))
            plt.plot(T, K, lw=2.5)
            plotMilestoneTemperatures()
            plt.xlabel('$T$', fontsize=24)
            plt.ylabel('$K$', fontsize=24)
            finalisePlot()

            plt.figure(figsize=(12, 8))
            plt.plot(T, peakFreq_primarysw, lw=2.5)
            plt.plot(T, peakFreq_secondarysw, lw=2.5)
            plt.plot(T, peakFreq_primaryturb, lw=2.5)
            plotMilestoneTemperatures()
            plt.xlabel('$T$', fontsize=24)
            plt.ylabel('$f_{\\mathrm{peak}}$', fontsize=24)
            plt.legend(['primarysw', 'secondarysw', 'primaryturb'], fontsize=20)
            finalisePlot()

            plt.figure(figsize=(12, 8))
            plt.plot(T, lenScale_primary, lw=2.5)
            plt.plot(T, lenScale_secondary, lw=2.5)
            plt.plot(T, lenScale_beta, lw=2.5)
            plotMilestoneTemperatures()
            plt.xlabel('$T$', fontsize=24)
            plt.ylabel('$\\mathrm{Length \\;\\; scale}$', fontsize=24)
            plt.legend(['primarysw', 'secondarysw', 'primary beta'], fontsize=20)
            plt.ylim(bottom=0, top=max(lenScale_primary[-1], lenScale_secondary[-1]))
            finalisePlot()

            plt.figure(figsize=(12, 8))
            plt.plot(T, Treh, lw=2.5)
            plt.plot(T, T, lw=2, ls='--')
            plotMilestoneTemperatures()
            plt.xlabel('$T$', fontsize=24)
            plt.ylabel('$T_{\\mathrm{reh}}$', fontsize=24)
            plt.ylim(bottom=0)
            finalisePlot()

            plt.figure(figsize=(12, 8))
            plt.plot(T, fluidVelocity, lw=2.5)
            plotMilestoneTemperatures()
            plt.xlabel('$T$', fontsize=24)
            plt.ylabel('$\\overline{U}_f$', fontsize=24)
            plt.ylim(bottom=0)
            finalisePlot()

            plt.figure(figsize=(12, 8))
            plt.plot(T, upsilon, lw=2.5)
            plotMilestoneTemperatures()
            plt.xlabel('$T$', fontsize=24)
            plt.ylabel('$\\Upsilon$', fontsize=24)
            plt.ylim(0, 1)
            finalisePlot()

            plt.figure(figsize=(12, 8))
            plt.plot(T, adiabaticIndex, lw=2.5)
            plotMilestoneTemperatures()
            plt.xlabel('$T$', fontsize=24)
            plt.ylabel('$\\Gamma$', fontsize=24)
            finalisePlot()

            plt.figure(figsize=(12, 8))
            plt.plot(T, csf, lw=2.5)
            plt.plot(T, cst, lw=2.5)
            plt.axhline(1/np.sqrt(3), lw=2, ls='--')
            plotMilestoneTemperatures()
            plt.xlabel('$T$', fontsize=24)
            plt.ylabel('$c_s$', fontsize=24)
            plt.legend(['false', 'true'], fontsize=20)
            finalisePlot()

            plt.figure(figsize=(12, 8))
            plt.plot(T, ndof, lw=2.5)
            plotMilestoneTemperatures()
            plt.xlabel('$T$', fontsize=24)
            plt.ylabel('$g_*$', fontsize=24)
            finalisePlot()

            plt.figure(figsize=(12, 8))
            plt.plot(T, redshift, lw=2.5)
            plotMilestoneTemperatures()
            plt.xlabel('$T$', fontsize=24)
            plt.ylabel('$\\mathcal{R}$', fontsize=24)
            finalisePlot()


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
    gwa = GWAnalyser(detectorClass, potentialClass, outputFolder)
    # Use this for scanning GWs and thermal params over temperature.
    gwa.determineGWs()
    gwa.scanGWs()
    # Use this for evaluating GWs using thermal params at the onset of percolation.
    #gwa.determineGWs()
    #hydroTester(potentialClass, outputFolder)


if __name__ == "__main__":
    main(LISA, RealScalarSingletModel, 'output/RSS/RSS_BP1/')
    #main(LISA, RealScalarSingletModel_HT, 'output/RSS_HT/RSS_HT_BP1/')
    #main(LISA, ToyModel, 'output/Toy/Toy_BP1/')
