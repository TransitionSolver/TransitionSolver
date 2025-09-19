"""
Analyse gravitational wave signals
==================================
"""

from __future__ import annotations

import os
import pathlib
import time
import traceback
import json
import copy
import logging
from typing import Callable, Type, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import scipy.optimize

from analysis import phase_structure
from analysis.phase_structure import Phase, PhaseStructure
from gws import kappa_nu_model, hydrodynamics, Detector, LISA, HydroVars
from models.real_scalar_singlet_model_boltz import RealScalarSingletModel_Boltz
from models.real_scalar_singlet_model_ht import RealScalarSingletModel_HT
from models.toy_model import ToyModel
from models.supercool_model import SMplusCubic
from models.analysable_potential import AnalysablePotential



GRAV_CONST = 6.7088e-39
logger = logging.getLogger(__name__)


class GWAnalysisSettings:
    bUseBubbleSeparation: bool
    bUseChapmanJouguetVelocity: bool
    sampleIndex: int
    suppliedRho_t: Optional[Callable[[float], float]]
    kappaColl: float
    kappaTurb: float

    def __init__(self):
        self.bUseBubbleSeparation = True
        self.bUseChapmanJouguetVelocity = True
        self.sampleIndex = -1
        self.suppliedRho_t = None
        self.kappaColl = 0.
        self.kappaTurb = 0.05


class AnalyseIndividualTransition:
    potential: AnalysablePotential
    phaseStructure: PhaseStructure
    fromPhase: Phase
    toPhase: Phase
    transitionReport: dict
    hydroVars: HydroVars
    hydroVarsReh: HydroVars
    detector: Detector
    peakAmplitude_sw_regular: float = 0.
    peakAmplitude_sw_soundShell: float = 0.
    peakFrequency_sw_bubbleSeparation: float = 0.
    peakFrequency_sw_shellThickness: float = 0.
    peakAmplitude_turb: float = 0.
    peakFrequency_turb: float = 0.
    peakAmplitude_coll: float = 0.
    peakFrequency_coll: float = 0.
    SNR: float = 0.
    vw: float = 0.
    T: float = 0.
    Treh: float = 0.
    H: float = 0.
    bubble_length_scale: float = 0.
    lengthScale_shellThickness: float = 0.
    adiabaticIndex: float = 0.
    fluidVelocity: float = 0.
    upsilon: float = 0.
    soundSpeed: float = 0.
    ndof: float = 0.
    redshift: float = 0.
    hStar: float = 0.
    turbNormalisationFactor: float = 1.
    beta: float = 0.
    kappaColl: float = 0.
    kappaTurb: float = 0.

    def __init__(self, phaseStructure: PhaseStructure, transitionReport: dict, potential: AnalysablePotential, detector:
            Detector):
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
        self.Treh = self.determineReheatingTemperature(settings)
        #self.hydroVars = hydrodynamics.getHydroVars(self.fromPhase, self.toPhase, self.potential, self.T)
        self.hydroVars = hydrodynamics.getHydroVars_new(self.fromPhase, self.toPhase, self.potential, self.T,
            self.phaseStructure.groundStateEnergyDensity)
        self.hydroVarsReh = hydrodynamics.getHydroVars(self.fromPhase, self.toPhase, self.potential, self.Treh)
        # Note setting vw via bubble_wall_velocity, also sets alpha
        # Todo:  refactor to make things more transparent
        self.vw = self.bubble_wall_velocity(settings.bUseChapmanJouguetVelocity)
        self.kappaColl = settings.kappaColl
        self.kappaTurb = settings.kappaTurb

        if self.kinetic_energy_fraction == 0:
            return 0., 0.
            
        self.bubble_length_scale = self.length_scale(settings.bUseBubbleSeparation, settings.sampleIndex)

        totalEnergyDensity = self.hydroVars.energyDensityFalse# - self.phaseStructure.groundStateEnergyDensity

        self.H = np.sqrt(8*np.pi*GRAV_CONST/3 * totalEnergyDensity)

        # Weight the enthalpy by the fraction of the Universe in each phase. This will underestimate the enthalpy
        # because we neglect the reheated regions around the bubble wall.
        # TODO: maybe we can get this from Giese's code?
        #averageEnthalpyDensity = self.hydroVars.enthalpyDensityFalse*0.71 + 0.29*self.hydroVars.enthalpyDensityTrue
        # Slightly better than averaging the enthalpy of each phase. Use energy conservation for the energy, and average
        # the pressure of each phase. Don't use totalEnergyDensity because we should not subtract off the ground state
        # energy density. The enthalpy is -T*dV/dT, so the ground state energy density doesn't contribute.
        Pf = 0.71 if settings.sampleIndex < 0 else self.transitionReport['Pf'][settings.sampleIndex]
        Pt = 1 - Pf
        averageEnthalpyDensity = self.hydroVars.pressureFalse*Pf + self.hydroVars.pressureTrue*Pt\
            + self.hydroVars.energyDensityFalse

        # Assuming energy conservation so averageEnergyDensity = totalEnergyDensity.
        self.adiabaticIndex = averageEnthalpyDensity / totalEnergyDensity

        self.fluidVelocity = np.sqrt(self.kinetic_energy_fraction/self.adiabaticIndex)
        # Assume the rotational modes are negligible.
        fluidVelocityLong = self.fluidVelocity

        #if not settings.bUseBubbleSeparation:
        #    settings.bUseBubbleSeparation = True
        #    lenscale = self.length_scale(settings)
        #    settings.bUseBubbleSeparation = False
        #    tau_sw = lenscale / fluidVelocityLong
        #else:
        tau_sw = self.bubble_length_scale / fluidVelocityLong
        self.upsilon = 1 - 1 / np.sqrt(1 + 2*self.H*tau_sw)

        self.soundSpeed = np.sqrt(self.hydroVars.soundSpeedSqFalse)
        #tau_c = bubble_length_scale / soundSpeed
        self.lengthScale_shellThickness = self.bubble_length_scale\
            * abs(self.vw - np.sqrt(self.hydroVars.soundSpeedSqFalse)) / self.vw

        self.ndof = self.potential.getDegreesOfFreedom(self.toPhase.findPhaseAtT(self.T, self.potential), self.T)
        ndofReh = self.potential.getDegreesOfFreedom(self.toPhase.findPhaseAtT(self.Treh, self.potential), self.Treh)
        #print('T_p:', self.T)
        #print('T_reh:', self.Treh)
        #print('g_*(T_p):', self.ndof)
        #print('g_*(T_reh):', ndofReh)
        self.redshiftAmp_radDom = 1.67e-5 * (100/self.ndof)**(1/3)

        # General form:
        #Omega_peak = redshift * K*K * upsilon * H*tau_c
        #print('Omega peak (general):', Omega_peak)

        self.redshiftFreq_radDom = 1.65e-5 * (self.Treh/100) * (ndofReh/100)**(1/6)
        self.redshiftAmp_radDom = 1.67e-5 * (100/ndofReh)**(1/3)
        # (s_0/s_*)^(1/3)
        KtoGeV = 8.617e-14
        GeVtoHz = 1.519e24
        T0 = 2.725 * KtoGeV
        Neff = 3.046
        g0 = 2 + 7/11*Neff
        s0 = 2*np.pi**2/45*g0*T0**3
        kmtoMpc = 3.241e-20
        #perSectoGeV = 6.582e-25
        hoverH0 = 1.0 / (100*kmtoMpc/GeVtoHz)
        s1 = self.hydroVarsReh.entropyDensityTrue
        H1 = self.H
        # a1/a0 = (s0/s1)^(1/3) and convert from GeV to Hz.
        self.redshiftFreq = (s0/s1)**(1/3) * GeVtoHz
        # (a1/a0)^4 (H0/H1)^2 = (s0/s1)^(4/3) * (H0/H1)^2, and absorb h^2 factor

        self.redshiftAmp = (s0/s1)**(4/3) * H1**2 * hoverH0**2
        #print('Redshift ratio (amp):', self.redshiftAmp / self.redshiftAmp_radDom)
        #print('Redshift ratio (freq):', self.redshiftFreq*self.H / self.redshiftFreq_radDom)

        #Omega_peak = 2.59e-6*(100/potential.ndof)**(1./3.) * K*K * H*(lenScale/(8*np.pi)**(1./3.))/soundSpeed * upsilon
        if self.kappaColl == 0:
            self.peakAmplitude_sw_regular = self.peak_amplitude_regular()
            zp = 10.  # This assumes the peak frequency corresponds to 10*lenScale. This result comes from simulations
            # (https://arxiv.org/pdf/1704.05871.pdf) and is expected to change if vw ~ vCJ (specifically zp will increase).
            #self.peakFrequency_sw_bubbleSeparation = 8.9e-6 * (self.ndof/100)**(1/6) * (self.Treh/100)\
            #    / (self.H*self.bubble_length_scale / ((8*np.pi)**(1/3) * self.vw)) * (zp/10)
            #self.peakFrequency_sw_shellThickness = 8.9e-6 * (self.ndof/100)**(1/6) * (self.Treh/100)\
            #    / (self.H*self.lengthScale_shellThickness / ((8*np.pi)**(1/3) * self.vw)) * (zp/10)
            self.peakFrequency_sw_bubbleSeparation = 1.58*self.redshiftFreq/self.bubble_length_scale*zp/10
            self.peakFrequency_sw_shellThickness = 1.58*self.redshiftFreq/self.lengthScale_shellThickness*zp/10

            #print('Alternative peak frequency:', 5.63e11 * (1/self.ndof)**(1/3) / (self.T*self.bubble_length_scale))
            #print('Alternative peak frequency (reheated):', 5.63e11 * (1/ndofReh)**(1/3) / (self.Treh*self.bubble_length_scale))

            #print('Conservation of energy peak frequency:', 2.23666e-38**(1/3)*1.519e24/self.bubble_length_scale/(self.hydroVars.entropyDensityTrue)**(1/3))
            #print('Conservation of energy peak frequency:', 4.28e11/self.bubble_length_scale/(self.hydroVars.entropyDensityTrue)**(1/3))

            self.peakAmplitude_sw_soundShell = self.peak_amplitude_sound_shell()

            #turbEfficiency = 0.05  # (1 - min(1, H*tau_sw))  # (1 - self.upsilon)

            self.peakAmplitude_turb = self.peak_amplitude_turb()
            #self.peakAmplitude_turb = 20.060 * self.redshift * (turbEfficiency*self.kinetic_energy_fraction)**(3/2)\
            #    * self.H*(self.bubble_length_scale / (8*np.pi)**(1/3))
            # I don't know where 2.7 comes from, it should be 2.9 (1.65 * 3.5/2).
            #self.peakFrequency_turb = 2.7e-5 * (self.ndof/100)**(1/6) * (self.Treh/100)\
            #    / (self.H*self.bubble_length_scale / (8*np.pi)**(1/3))
            self.peakFrequency_turb = 3.5*self.redshiftFreq / self.bubble_length_scale

            # The spectral shape is multiplied by this normalisation factor. However, the normalisation factor is initially
            # set to 1 upon the construction of this instance of the class. Use the spectral shape at the peak frequency to
            # determine the normalisation factor.
            self.turbNormalisationFactor = 1 / self.spectral_shape_turb(self.peakFrequency_turb)
            # Then make sure the peak amplitude is corrected to match the peak amplitude of the GW signal.
            self.peakAmplitude_turb /= self.turbNormalisationFactor
        else:
            self.peakAmplitude_coll = self.peak_amplitude_coll()
            self.beta = (8*np.pi)**(1/3) * self.vw / self.bubble_length_scale
            #self.peakFrequency_coll = (0.77*self.beta/(2*np.pi))/self.beta * (self.beta/self.H) * self.hStar
            #self.peakFrequency_coll = 0.77 * self.beta / (2*np.pi) * self.hStar / self.H
            self.peakFrequency_coll = self.redshiftFreq * (0.77 * (8*np.pi)**(1/3) * self.vw
                / (2*np.pi * self.bubble_length_scale))
            #print('Alternative peak frequency (coll):', 0.77 * self.beta / (2*np.pi) * 4.28e11/(self.hydroVars.entropyDensityTrue)**(1/3))

    def peak_amplitude_regular(self) -> float:
        """
        Fit from https://arxiv.org/pdf/1704.05871 taking account of erratum
        """
        Omega_gw = 0.012
        return 2.061*Omega_gw*self.redshiftAmp * self.kinetic_energy_fraction*self.kinetic_energy_fraction * (self.H* self.bubble_length_scale / self.soundSpeed) * self.upsilon

    def peak_amplitude_sound_shell(self) -> float:
        """
        Based on https://arxiv.org/pdf/1909.10040.pdf.
        """
        # Omega_gw = 0.01  # From https://arxiv.org/pdf/1704.05871.pdf TABLE IV.
        Omega_gw = 0.012  # From erratum of https://arxiv.org/pdf/1704.05871.pdf TABLE IV.
        rb = self.peakFrequency_sw_bubbleSeparation / self.peakFrequency_sw_shellThickness
        mu_f = 4.78 - 6.27*rb + 3.34*rb**2
        Am = 3*self.kinetic_energy_fraction**2*Omega_gw / mu_f
        # Roughly a factor of 0.2 smaller than the regular sound wave peak amplitude. Coming from 3*Omega_gw/0.15509
        # * (1/mu_f) * (8pi)^(1/3). For vw ~ 1 and soundSpeed ~ 1/sqrt(3), mu_f ~ 0.4.
        return (self.H*self.bubble_length_scale / self.soundSpeed) * Am * self.redshiftAmp * self.upsilon

    def peak_amplitude_coll(self) -> float:
        """
        Based on https://arxiv.org/pdf/2208.11697.pdf
        """
        A = 5.13e-2
        return A * self.redshiftAmp * (self.H*self.bubble_length_scale / ((8*np.pi)**(1/3) * self.vw))**2 * self.kinetic_energy_fraction**2

    def peak_amplitude_turb(self) -> float:
        return 9.0*self.redshiftAmp * (self.H*self.bubble_length_scale) * (self.kappaTurb*self.kinetic_energy_fraction)**(3/2)

    def SNR(self, amplitude_func: Callable[[float], float], a=1e-11, b=1e3) -> float:
        """
        @returns Signal to noise ratio for detector and given signal
        """
        def integrand(log_f):
            f = np.exp(log_f)
            return f * (amplitude_func(f) / self.detector(f))**2

        integral = scipy.integrate.quad(integrand, np.log(a), np.log(b))[0]
        return np.sqrt(self.detector.detection_time * self.detector.channels * integral)

    def spectral_shape_sw(self, f: float) -> float:
        x = f / self.peakFrequency_sw_bubbleSeparation
        return x**3 * (7 / (4 + 3*x**2))**3.5

    def spectral_shape_sw_double_broken(self, f: float):
        if self.kinetic_energy_fraction == 0:
            return 0.

        # From https://arxiv.org/pdf/2209.13551.pdf (Eq. 2.11), originally from https://arxiv.org/pdf/1909.10040.pdf
        # (Eq. # 5.7).
        b = 1
        rb = self.peakFrequency_sw_bubbleSeparation / self.peakFrequency_sw_shellThickness
        m = (9*rb**4 + b) / (rb**4 + 1)
        x = f/self.peakFrequency_sw_shellThickness
        return x**9 * ((1 + rb**4) / (rb**4 + x**4))**((9 - b) / 4) * ((b + 4) / (b + 4 - m + m*x**2))**((b + 4) / 2)

    def spectral_shape_turb(self, f: float) -> float:
        x = f / self.peakFrequency_turb
        #return self.turbNormalisationFactor * x**3 / ((1 + x)**(11/3) * (1 + 8*np.pi*f/self.hStar))
        return self.turbNormalisationFactor * x**3 / ((1 + x)**(11/3) * (1 + 8*np.pi*f/(self.redshiftFreq*self.H)))

    def spectral_shape_coll(self, f):
        if self.kinetic_energy_fraction == 0:
            return np.zeros_like(f)

        a = 2.41
        b = 2.42
        c = 4.08
        fp0 = self.peakFrequency_coll
        # Using normalised spectral shape, so A = 5.13e-2 is moved to the amplitude calculation.
        return (a+b)**c/(b*(f/fp0)**(-a/c) + a*(f/fp0)**(b/c))**c

    def get_gw_total_func(self, soundShell: bool = True) -> Callable[[float], float]:
        if self.kappaColl > 0:
            return lambda f: self.peakAmplitude_coll*self.spectral_shape_coll(f)

        if self.kinetic_energy_fraction == 0:
            return np.zeros_like

        if soundShell:
            return lambda f: self.peakAmplitude_sw_soundShell*self.spectral_shape_sw_double_broken(f) + self.peakAmplitude_turb*self.spectral_shape_turb(f)

        return lambda f: self.peakAmplitude_sw_regular*self.spectral_shape_sw(f) + self.peakAmplitude_turb*self.spectral_shape_turb(f)

    def get_gw_sw_func(self, soundShell: bool = False) -> Callable[[float], float]:
        if self.kappaColl > 0:
            return np.zeros_like

        if soundShell:
            return lambda f: self.peakAmplitude_sw_soundShell*self.spectral_shape_sw_double_broken(f)

        return lambda f: self.peakAmplitude_sw_regular*self.spectral_shape_sw(f)

    def get_gw_turb_func(self) -> Callable[[float], float]:
        if self.kappaColl > 0:
            return np.zeros_like

        if self.peakAmplitude_turb == 0. or self.peakFrequency_turb == 0.:
            return np.zeros_like

        return lambda f: self.peakAmplitude_turb*self.spectral_shape_turb(f)

    def get_gw_coll_func(self) -> Callable[[float], float]:
        if self.peakAmplitude_coll == 0. or self.peakFrequency_coll == 0.:
            return np.zeros_like

        return lambda f: self.peakAmplitude_coll*self.spectral_shape_coll(f)

    def determineTransitionTemperature(self, settings: GWAnalysisSettings) -> float:
        if settings.sampleIndex < 0:
            return self.transitionReport['Tp']

        return self.transitionReport['TSubsample'][settings.sampleIndex]

    def determineReheatingTemperature(self, settings: GWAnalysisSettings) -> float:
        if settings.sampleIndex < 0:
            return self.transitionReport['Treh_p']

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

    def bubble_wall_velocity(self, use_cj) -> float:
        """
        Using either Chapman-Jouguet hydrodynamical estimate or value from transition report
        
        @returns Bubble wall velocity
        """
        if not use_cj:
            return self.transitionReport['vw']
        return self.hydroVars.cj_velocity

    @property
    def kappa_sound(self) -> float:
        """
        @returns Efficiency of sound waves using the kappa-nu model
        """
        # adjust the vw value to avoid numerical instabilities
        
        vw = self.vw
        
        if vw <=0.:
            vw = 1e-6

        if vw > 0.999999:
            vw = 0.999999
            
        if vw != self.vw:
            logger.warn(f"vw adjusted from {self.vw} to {vw} to avoid numerical instability")

        kappa_sound = kappa_nu_model(self.hydroVars.soundSpeedSqTrue, self.hydroVars.alpha, vw)

        if kappa_sound > 1:
            raise RuntimeError("kappa_sound > 1: {kappa_sound}")

        if kappa_sound <= 0:
            raise RuntimeError("kappa_sound <= 0: {kappa_sound}")

        return kappa_sound

    @property
    def kinetic_energy_fraction(self) -> float:
        """
        kappa from collisions if > 0 else sound

        @returns Kinetic energy fraction
        """    
        if self.hydroVars.soundSpeedSqTrue <= 0:
            return 0.

        factor = self.kappaColl if self.kappaColl > 0 else self.kappa_sound
        K = factor * self.hydroVars.K

        if K > 1:
            logger.warn(f'K > 1: {K}')

        if K < 0:
            raise RuntimeError("K < 0: {K}")

        return K

    def length_scale(self, use_bubble_separation, sample_index) -> float:
        """
        @returns Characteristic bubble length scale
        """
        key = 'meanBubbleSeparation' if use_bubble_separation else 'meanBubbleRadius'
        scale = self.transitionReport[key]
        if sample_index < 0:
            return scale
        return scale[sample_index]


class GWAnalyser:
    detector: Optional[Detector]
    potential: AnalysablePotential
    phaseHistoryReport: dict
    relevantTransitions: list[dict]

    def __init__(self, detectorClass: Type[Detector], potentialClass=None, outputFolder=None,
            bForceAllTransitionsRelevant: bool = False, phase_history=None, phase_structure_file=None, potential=None):
        self.detector = detectorClass()
        
        if phase_history is None:
            phase_history_file = os.path.join(outputFolder, 'phase_history.json')

            with open(phase_history_file , 'r') as f:
                self.phaseHistoryReport = json.load(f)
        else:
            self.phaseHistoryReport = phase_history

        if phase_structure_file is None:
            phase_structure_file = os.path.join(outputFolder, 'phase_structure.dat')

        bSuccess, self.phaseStructure = phase_structure.load_data(phase_structure_file)

        if not bSuccess:
            raise Exception("Failed to load phase structure, please check this file exists in the relevant directory.")

        if potential is None:
            parameter_point_file = os.path.join(outputFolder, 'parameter_point.txt')
            parameter_point = np.loadtxt(parameter_point_file)
            bUseBoltzmannSuppression = potentialClass == SMplusCubic
            self.potential = potentialClass(*parameter_point, bUseBoltzmannSuppression=bUseBoltzmannSuppression)
        else:
            self.potential = potential
                     
        self.relevantTransitions = extractRelevantTransitions(self.phaseHistoryReport,
            bForceAllTransitionsRelevant=bForceAllTransitionsRelevant)

        if not self.relevantTransitions:
            raise Exception('No relevant transition detected in the phase history')

    def report(self, settings: GWAnalysisSettings = None):
        """
        @returns Data on GW spectrum
        """
        if settings is None:
            settings = GWAnalysisSettings()
        
        report = {}
        
        for transition in self.relevantTransitions:
            gws = AnalyseIndividualTransition(self.phaseStructure, transition, self.potential, self.detector)
            gws.determineGWs(settings)
            sound_shell = gws.get_gw_total_func(soundShell=True)
            
            spectra = report[transition['id']] = {}
            spectra['Peak amplitude (sw, sound shell)'] = gws.peakAmplitude_sw_soundShell
            spectra['Peak frequency (sw, bubble separation)'] = gws.peakFrequency_sw_bubbleSeparation
            spectra['Peak frequency (sw, shell thickness)'] = gws.peakFrequency_sw_shellThickness
            spectra['Peak amplitude (turb)'] = gws.peakAmplitude_turb
            spectra['Peak frequency (turb)'] = gws.peakFrequency_turb
            spectra['Peak amplitude (coll)'] = gws.peakAmplitude_coll
            spectra['Peak frequency (coll)'] = gws.peakFrequency_coll
            spectra['SNR (double)'] = gws.SNR(sound_shell)

        return report

    def plot(self, settings: GWAnalysisSettings = None, show=False, frequencies=None):
        """
        @returns Figure of plot of data on GW spectrum
        """
        if settings is None:
            settings = GWAnalysisSettings()
            
        if frequencies is None:
            frequencies = np.logspace(-11, 3, 1000)
        
        n = len(self.relevantTransitions)
        fig, ax = plt.subplots(n)
        
        if n <= 1:
            ax = [ax]
        
        for a, transition in zip(ax, self.relevantTransitions):

            gws = AnalyseIndividualTransition(self.phaseStructure, transition, self.potential, self.detector)
            gws.determineGWs(settings)
            total = gws.get_gw_total_func(soundShell=True)
            sw = gws.get_gw_sw_func(soundShell=True)
            turb = gws.get_gw_turb_func()
            coll = gws.get_gw_coll_func()

            a.loglog(frequencies, self.detector(frequencies), label="noise")
            a.loglog(frequencies, total(frequencies), zorder=10, label="total")
            a.loglog(frequencies, sw(frequencies), label="sw")
            a.loglog(frequencies, turb(frequencies), label="turb")
            a.loglog(frequencies, coll(frequencies), label="coll")
            a.legend()
        
        if show:
            plt.show()
        
        return fig


    def determineGWs(self, GWsOutputFolder, settings: GWAnalysisSettings = None):
        # If no settings are supplied, use default settings.
        if settings is None:
            settings = GWAnalysisSettings()
        for transitionReport in self.relevantTransitions:
            gws = AnalyseIndividualTransition(self.phaseStructure, transitionReport, self.potential,
                self.detector)
            gws.determineGWs(settings)
            #gwFunc_regular = gws.get_gw_total_func(soundShell=False)
            gwFunc_soundShell = gws.get_gw_total_func(soundShell=True)
            #SNR_single = gws.SNR(gwFunc_regular)
            SNR_double = gws.SNR(gwFunc_soundShell)
            print('Transition ID:', transitionReport['id'])
            #print('Peak amplitude (sw, regular):', gws.peakAmplitude_sw_regular)
            print('Peak amplitude (sw, sound shell):', gws.peakAmplitude_sw_soundShell)
            print('Peak frequency (sw, bubble separation):', gws.peakFrequency_sw_bubbleSeparation)
            print('Peak frequency (sw, shell thickness):', gws.peakFrequency_sw_shellThickness)
            print('Peak amplitude (turb):', gws.peakAmplitude_turb)
            print('Peak frequency (turb):', gws.peakFrequency_turb)
            print('Peak amplitude (coll):', gws.peakAmplitude_coll)
            print('Peak frequency (coll):', gws.peakFrequency_coll)
            #print('SNR (single):', SNR_single)
            print('SNR (double):', SNR_double)

            #gwFunc_sw_regular = gws.get_gw_sw_func(soundShell=False)
            gwFunc_sw_soundShell = gws.get_gw_sw_func(soundShell=True)
            gwFunc_turb = gws.get_gw_turb_func()
            gwFunc_coll = gws.get_gw_coll_func()

            plt.loglog(self.detector.sensitivityCurve[0], self.detector.sensitivityCurve[1])
            #plt.loglog(self.detector.sensitivityCurve[0], np.array([gwFunc_regular(f) for f in
            #    self.detector.sensitivityCurve[0]]))
            plt.loglog(self.detector.sensitivityCurve[0], np.array([gwFunc_soundShell(f) for f in
                self.detector.sensitivityCurve[0]]), zorder=10)
            #plt.loglog(self.detector.sensitivityCurve[0], np.array([gwFunc_sw_regular(f) for f in
            #    self.detector.sensitivityCurve[0]]))
            plt.loglog(self.detector.sensitivityCurve[0], np.array([gwFunc_sw_soundShell(f) for f in
                self.detector.sensitivityCurve[0]]))
            plt.loglog(self.detector.sensitivityCurve[0], np.array([gwFunc_turb(f) for f in
                self.detector.sensitivityCurve[0]]))
            plt.loglog(self.detector.sensitivityCurve[0], np.array([gwFunc_coll(f) for f in
                self.detector.sensitivityCurve[0]]))
            #plt.legend(['noise', 'total (single)', 'total (double)', 'sw (single)', 'sw (double)', 'turb'])
            plt.legend(['noise', 'total', 'sw', 'turb', 'coll'])
            plt.ylim(bottom=1e-20, top=1e5)
            plt.margins(0, 0)
            plt.savefig(GWsOutputFolder+'GWs_SignalVsSensitiviy.pdf')

    def determineGWs_withColl(self, GWsOutputFolder, file_name=None, settings: GWAnalysisSettings = None):
         # If no settings are supplied, use default settings.
        if settings is None:
            settings = GWAnalysisSettings()
        for transitionReport in self.relevantTransitions:
            gws_noColl = AnalyseIndividualTransition(self.phaseStructure, transitionReport, self.potential,
                self.detector)
            gws_coll = AnalyseIndividualTransition(self.phaseStructure, transitionReport, self.potential,
                self.detector)
            # make two copies of settings as we need to modify them from user input in this method
            # maybe a mild design flaw to do that, but its fairly transparent here as point is to run
            # with two different kapaa_coll
            settings_coll = copy.copy(settings)
            settings_nocoll = copy.copy(settings)
            # set kappacoll setting for each case (with collions only and without collsions)
            settings_coll.kappaColl = 1.
            settings_nocoll.kappaColl = 0
            gws_noColl.determineGWs(settings_nocoll)
            gws_coll.determineGWs(settings_coll)
            gwFunc_noColl = gws_noColl.get_gw_total_func(soundShell=True)
            gwFunc_coll = gws_coll.get_gw_total_func()
            SNR_noColl = gws_noColl.SNR(gwFunc_noColl)
            SNR_coll = gws_noColl.SNR(gwFunc_coll)
            print('Transition ID:', transitionReport['id'])
            #print('Peak amplitude (sw, regular):', gws.peakAmplitude_sw_regular)
            print('Peak amplitude (sw, sound shell):', gws_noColl.peakAmplitude_sw_soundShell)
            print('Peak frequency (sw, bubble separation):', gws_noColl.peakFrequency_sw_bubbleSeparation)
            print('Peak frequency (sw, shell thickness):', gws_noColl.peakFrequency_sw_shellThickness)
            print('Peak amplitude (turb):', gws_noColl.peakAmplitude_turb)
            print('Peak frequency (turb):', gws_noColl.peakFrequency_turb)
            print('Peak amplitude (coll):', gws_coll.peakAmplitude_coll)
            print('Peak frequency (coll):', gws_coll.peakFrequency_coll)
            #print('SNR (single):', SNR_single)
            print('SNR (no coll):', SNR_noColl)
            print('SNR (coll):', SNR_coll)

            gwFunc_sw = gws_noColl.get_gw_sw_func(soundShell=True)
            gwFunc_turb = gws_noColl.get_gw_turb_func()
            gwFunc_coll = gws_coll.get_gw_coll_func()

            plt.rcParams["text.usetex"] = True

            if file_name:
                with open(file_name, 'w') as file:
                    for i in range(len(self.detector.sensitivityCurve[0])):
                        f = self.detector.sensitivityCurve[0][i]
                        tot = gwFunc_noColl(f)
                        sw = gwFunc_sw(f)
                        turb = gwFunc_turb(f)
                        coll = gwFunc_coll(f)
                        file.write(' '.join([str(f), str(tot), str(sw), str(turb), str(coll)]) + '\n')
                return

            plt.figure(figsize=(12,8))
            #plt.loglog(self.detector.sensitivityCurve[0], self.detector.sensitivityCurve[1])
            plt.loglog(self.detector.sensitivityCurve[0], np.array([gwFunc_noColl(f) for f in
                self.detector.sensitivityCurve[0]]), lw=2.5, zorder=10)
            #plt.loglog(self.detector.sensitivityCurve[0], np.array([gwFunc_sw_regular(f) for f in
            #    self.detector.sensitivityCurve[0]]))
            plt.loglog(self.detector.sensitivityCurve[0], np.array([gwFunc_sw(f) for f in
                self.detector.sensitivityCurve[0]]), lw=2.5)
            plt.loglog(self.detector.sensitivityCurve[0], np.array([gwFunc_turb(f) for f in
                self.detector.sensitivityCurve[0]]), lw=2.5)
            plt.loglog(self.detector.sensitivityCurve[0], np.array([gwFunc_coll(f) for f in
                self.detector.sensitivityCurve[0]]), lw=2.5)
            #plt.legend(['noise', 'fluid', 'sw', 'turb', 'coll'])
            plt.xlabel('$f \\;\\; [\\mathrm{Hz}]$', fontsize=40)
            plt.ylabel('$h^2 \\Omega_{\\mathrm{GW}}(f)$', fontsize=40)
            plt.legend(['$\\mathrm{fluid}$', '$\\mathrm{sound \\; waves}$', '$\\mathrm{turbulence}$',
                '$\\mathrm{collisions}$'], fontsize=32)
            plt.ylim(bottom=1e-20)
            plt.tick_params(size=8, labelsize=28)
            plt.margins(0, 0)
            plt.tight_layout(pad=1.8)
            pathlib.Path(str(pathlib.Path('output/plots'))).mkdir(parents=True, exist_ok=True)
            plt.savefig(GWsOutputFolder+'GWs_SignalVsSensitiviy.pdf')
            #plt.show()

    def getHackMeanBubbleRadius(self, transitionReport: dict):
        T = transitionReport['TSubsample']
        Tcoarse = transitionReport['T']
        H = transitionReport['H']
        Pf = transitionReport['Pf']
        SonTcoarse = transitionReport['SonT']
        interpSonT = scipy.interpolate.interp1d(Tcoarse, SonTcoarse)
        SonT = interpSonT(T)
        Gamma = [T[i]**4 * (SonT[i]/(2*np.pi))**(3/2) * np.exp(-SonT[i]) for i in range(len(T))]
        vw = [1.]*(len(T))  # TODO: fix this!

        n = [0]*len(T)

        for i in range(len(T)):
            integrand = [0]*(len(T) - i)

            for j in range(i, len(T)):
                integrand[j-i] = Gamma[j] * Pf[j] / (T[j]**4 * H[j])

            integral = 0

            for j in range(1, len(integrand)):
                integral += 0.5*(integrand[j] + integrand[j-1])

            n[i] = T[i]**3 * integral

        meanBubbleRadius = [0]*len(T)

        for i in range(len(T)):
            outerIntegrand = [0]*(len(T) - i)

            for j in range(i, len(T)):
                innerIntegrand = [0]*(j - i)

                for k in range(i, j):
                    innerIntegrand[k-i] = vw[k] / H[k]

                innerIntegral = 0

                for k in range(1, len(innerIntegrand)):
                    innerIntegral += 0.5*(innerIntegrand[k] + innerIntegrand[k-1])

                outerIntegrand[j-i] += innerIntegral * (Gamma[j] * Pf[j]) / (T[j] * H[j])

            outerIntegral = 0

            for j in range(1, len(outerIntegrand)):
                outerIntegral += 0.5*(outerIntegrand[j] + outerIntegrand[j-1])

            meanBubbleRadius[i] = T[i]**2/n[i] * outerIntegral if n[i] > 0 else meanBubbleRadius[i-1]

        return meanBubbleRadius

    def scanGWs(self, saveFolderName: str = '', bCombined: bool = False):
        if saveFolderName != '':
            pathlib.Path(str(pathlib.Path(saveFolderName))).mkdir(parents=True, exist_ok=True)
        for transitionReport in self.relevantTransitions:
            transitionID = transitionReport['id']
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

            # Increase this to reduce the number of samples used in the plots (to improve runtime while getting the
            # general form of the results). If skipFactor=3 (for example), every third sample will be used.
            skipFactor = 1
            for i in range(11, min(len(allT)//skipFactor-1,len(allT))):
                indices.append(1+i*skipFactor)

            #print('Sampling indices:', indices)

            """Ttemp = transitionReport['TSubsample']
            hackMeanBubbleRadiusArray: List[float] = self.getHackMeanBubbleRadius(transitionReport)
            prevMeanBubbleRadiusArray: List[float] = transitionReport['meanBubbleRadiusArray']
            meanBubbleSeparationArray: List[float] = transitionReport['meanBubbleSeparationArray']

            plt.plot(Ttemp, hackMeanBubbleRadiusArray)
            plt.plot(Ttemp, prevMeanBubbleRadiusArray)
            plt.plot(Ttemp, meanBubbleSeparationArray)
            #plt.xscale('log')
            plt.yscale('log')
            plt.margins(0., 0.)
            plt.show()"""

            peakAmplitude_sw_regular: List[float] = []
            peakAmplitude_sw_soundShell: List[float] = []
            peakFrequency_sw_bubbleSeparation: List[float] = []
            peakFrequency_sw_shellThickness: List[float] = []
            peakAmplitude_turb: List[float] = []
            peakFrequency_turb: List[float] = []
            SNR_regular: List[float] = []
            SNR_soundShell: List[float] = []
            K: List[float] = []
            kappa: List[float] = []
            kappa_original: List[float] = []
            alpha: List[float] = []
            vw: List[float] = []
            Treh: List[float] = []
            TrehApprox: List[float] = []
            bubble_length_scale: List[float] = []
            lengthScale_shellThickness: List[float] = []
            meanBubbleRadius: List[float] = []
            adiabaticIndex: List[float] = []
            fluidVelocity: List[float] = []
            upsilon: List[float] = []
            csfSq: List[float] = []
            cstSq: List[float] = []
            csf: List[float] = []
            cst: List[float] = []
            ndof: List[float] = []
            redshiftAmp: List[float] = []
            redshiftFreq: List[float] = []
            beta: List[float] = []
            lengthScale_beta: List[float] = []
            betaV: float = transitionReport.get('betaV', 0.)
            GammaMax: float = transitionReport.get('GammaMax', 0.)
            lengthScale_betaV: float = (np.sqrt(2*np.pi)*GammaMax/betaV)**(-1/3) if betaV > 0. else 0.

            #if bCombined:
            peakAmplitude_sw_regular_2: List[float] = []
            peakAmplitude_sw_soundShell_2: List[float] = []
            peakFrequency_sw_bubbleSeparation_2: List[float] = []
            peakFrequency_sw_shellThickness_2: List[float] = []
            peakAmplitude_turb_2: List[float] = []
            peakFrequency_turb_2: List[float] = []
            SNR_regular_2: List[float] = []
            SNR_soundShell_2: List[float] = []
            K_2: List[float] = []
            kappa_2: List[float] = []
            kappa_original_2: List[float] = []
            alpha_2: List[float] = []
            vw_2: List[float] = []
            Treh_2: List[float] = []
            TrehApprox_2: List[float] = []
            bubble_length_scale_2: List[float] = []
            lengthScale_shellThickness_2: List[float] = []
            meanBubbleRadius_2: List[float] = []
            adiabaticIndex_2: List[float] = []
            fluidVelocity_2: List[float] = []
            upsilon_2: List[float] = []
            csfSq_2: List[float] = []
            cstSq_2: List[float] = []
            csf_2: List[float] = []
            cst_2: List[float] = []
            ndof_2: List[float] = []
            redshiftAmp_2: List[float] = []
            redshiftFreq_2: List[float] = []
            beta_2: List[float] = []
            lengthScale_beta_2: List[float] = []
            betaV_2: float = transitionReport.get('betaV', 0.)
            GammaMax_2: float = transitionReport.get('GammaMax', 0.)
            lengthScale_betaV_2: float = (np.sqrt(2*np.pi)*GammaMax/betaV)**(-1/3) if betaV > 0. else 0.

            for i in indices:
                #if allT[i] > 83.87:
                #    continue
                gws = AnalyseIndividualTransition(self.phaseStructure, transitionReport, self.potential,
                    self.detector)
                settings = GWAnalysisSettings()
                settings.sampleIndex = i
                settings.suppliedRho_t = rhot_interp
                settings.bUseBubbleSeparation = True
                try:
                    gws.determineGWs(settings)
                except:
                    print('Failed for index:', i)
                    print('Temperature:', allT[i])
                    continue

                if gws.peakAmplitude_sw_regular > 0 and gws.peakAmplitude_turb > 0:
                    gwFunc_regular = gws.get_gw_total_func(soundShell=False)
                    gwFunc_soundShell = gws.get_gw_total_func(soundShell=True)

                    if abs(gws.hydroVars.soundSpeedSqTrue - 0.333) > 0.03:
                        continue

                    T.append(allT[i])
                    peakAmplitude_sw_regular.append(gws.peakAmplitude_sw_regular)
                    peakAmplitude_sw_soundShell.append(gws.peakAmplitude_sw_soundShell)
                    peakFrequency_sw_bubbleSeparation.append(gws.peakFrequency_sw_bubbleSeparation)
                    peakFrequency_sw_shellThickness.append(gws.peakFrequency_sw_shellThickness)
                    peakAmplitude_turb.append(gws.peakAmplitude_turb)
                    peakFrequency_turb.append(gws.peakFrequency_turb)
                    SNR_regular.append(gws.SNR(gwFunc_regular))
                    SNR_soundShell.append(gws.SNR(gwFunc_soundShell))
                    K.append(gws.K)
                    kappa.append(gws.kappa_sound())
                    alpha.append(gws.alpha)
                    vw.append(gws.vw)
                    Treh.append(gws.Treh)
                    TrehApprox.append(T[-1]*(1+alpha[-1])**(1/4))
                    bubble_length_scale.append(gws.bubble_length_scale)
                    lengthScale_shellThickness.append(gws.lengthScale_shellThickness)
                    meanBubbleRadius.append(transitionReport['meanBubbleRadiusArray'][i])
                    adiabaticIndex.append(gws.adiabaticIndex)
                    fluidVelocity.append(gws.fluidVelocity)
                    upsilon.append(gws.upsilon)
                    csfSq.append(gws.hydroVars.soundSpeedSqFalse)
                    cstSq.append(gws.hydroVars.soundSpeedSqTrue)
                    csf.append(np.sqrt(csfSq[-1]))
                    cst.append(np.sqrt(cstSq[-1]))
                    ndof.append(gws.ndof)
                    redshiftAmp.append(gws.redshiftAmp)
                    redshiftFreq.append(gws.redshiftFreq)
                    beta.append(transitionReport['beta'][i])
                    lengthScale_beta.append((8*np.pi)**(1/3) * vw[-1] / beta[-1])

                if bCombined:
                    gws = AnalyseIndividualTransition(self.phaseStructure, transitionReport, self.potential,
                        self.detector)
                    settings = GWAnalysisSettings()
                    settings.sampleIndex = i
                    settings.suppliedRho_t = rhot_interp
                    settings.bUseBubbleSeparation = False
                    try:
                        gws.determineGWs(settings)
                    except:
                        print('Failed for index:', i)
                        print('Temperature:', allT[i])
                        continue

                    if gws.peakAmplitude_sw_regular > 0 and gws.peakAmplitude_turb > 0:
                        gwFunc_regular = gws.get_gw_total_func(soundShell=False)
                        gwFunc_soundShell = gws.get_gw_total_func(soundShell=True)

                        #T.append(allT[i])
                        peakAmplitude_sw_regular_2.append(gws.peakAmplitude_sw_regular)
                        peakAmplitude_sw_soundShell_2.append(gws.peakAmplitude_sw_soundShell)
                        peakFrequency_sw_bubbleSeparation_2.append(gws.peakFrequency_sw_bubbleSeparation)
                        peakFrequency_sw_shellThickness_2.append(gws.peakFrequency_sw_shellThickness)
                        peakAmplitude_turb_2.append(gws.peakAmplitude_turb)
                        peakFrequency_turb_2.append(gws.peakFrequency_turb)
                        SNR_regular_2.append(gws.SNR(gwFunc_regular))
                        SNR_soundShell_2.append(gws.SNR(gwFunc_soundShell))
                        K_2.append(gws.K)
                        kappa_2.append(gws.kappa_sound())
                        alpha_2.append(gws.alpha)
                        vw_2.append(gws.vw)
                        Treh_2.append(gws.Treh)
                        TrehApprox_2.append(T[-1]*(1+alpha[-1])**(1/4))
                        bubble_length_scale_2.append(gws.bubble_length_scale)
                        lengthScale_shellThickness_2.append(gws.lengthScale_shellThickness)
                        meanBubbleRadius_2.append(transitionReport['meanBubbleRadiusArray'][i])
                        adiabaticIndex_2.append(gws.adiabaticIndex)
                        fluidVelocity_2.append(gws.fluidVelocity)
                        upsilon_2.append(gws.upsilon)
                        csfSq_2.append(gws.hydroVars.soundSpeedSqFalse)
                        cstSq_2.append(gws.hydroVars.soundSpeedSqTrue)
                        csf_2.append(np.sqrt(csfSq[-1]))
                        cst_2.append(np.sqrt(cstSq[-1]))
                        ndof_2.append(gws.ndof)
                        redshiftAmp_2.append(gws.redshiftAmp)
                        redshiftFreq_2.append(gws.redshiftFreq)
                        beta_2.append(transitionReport['beta'][i])
                        lengthScale_beta_2.append((8*np.pi)**(1/3) * vw[-1] / beta[-1])

            print('Analysis took:', time.perf_counter() - startTime, 'seconds')

            def plotMilestoneTemperatures():
                if 'Tn' in transitionReport: plt.axvline(transitionReport['Tn'], ls='--', c='r')
                if 'TGammaMax' in transitionReport: plt.axvline(transitionReport['TGammaMax'], ls='--', c='m')
                if 'Tp' in transitionReport: plt.axvline(transitionReport['Tp'], ls='--', c='g')
                if 'Te' in transitionReport: plt.axvline(transitionReport['Te'], ls='--', c='b')
                if 'Tf' in transitionReport: plt.axvline(transitionReport['Tf'], ls='--', c='k')

            def finalisePlot(plotName: str = 'plot'):
                plt.tick_params(size=8, labelsize=28)
                plt.margins(0, 0)
                plt.tight_layout()
                if saveFolderName != '':
                    plt.savefig(saveFolderName+plotName+'.pdf', bbox_inches='tight')
                else:
                    plt.show()

            plt.rcParams["text.usetex"] = True

            # Used to control the colour of lines, e.g. allowing the fourth plotted line to have the same colour as the
            # first plotted line.
            cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

            import matplotlib.lines as mlines

            if not bCombined:
                plt.figure(figsize=(12, 8))
                plt.plot(self.detector.sensitivityCurve[0], self.detector.sensitivityCurve[1], lw=2.5, label='_nolegend_')
                plt.scatter(peakFrequency_sw_shellThickness, peakAmplitude_sw_soundShell, c=T, cmap='coolwarm', marker='o',
                    s=49)
                plt.scatter(peakFrequency_turb, peakAmplitude_turb, c=T, cmap='coolwarm', marker='x', s=49)
                plt.xscale('log')
                plt.yscale('log')
                plt.xlim(left=min(peakFrequency_sw_shellThickness[0], peakFrequency_turb[0]), right=max(0.1,
                    peakFrequency_sw_shellThickness[-1], peakFrequency_turb[-1]))
                plt.ylim(top=100)
                plt.xlabel('$f_{\\mathrm{peak}} \\;\\; \\mathrm{[Hz]}$', fontsize=40)
                plt.ylabel('$h^2 \\Omega_{\\mathrm{peak}}$', fontsize=40)
                colorbar = plt.colorbar()
                colorbar.set_label(label='$T_* \\;\\; \\mathrm{[GeV]}$', fontsize=28)
                colorbar.ax.tick_params(labelsize=16)

                # See https://stackoverflow.com/a/47392973
                soundWaves = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10,
                    label='$\\mathrm{sound \\; waves}$')
                turbulence = mlines.Line2D([], [], color='black', marker='x', linestyle='None', markersize=10,
                    label='$\\mathrm{turbulence}$')
                plt.legend(handles=[soundWaves, turbulence], fontsize=28, loc='lower left', handletextpad=0.1)
            else:
                plt.figure(figsize=(12, 8))
                plt.plot(self.detector.sensitivityCurve[0], self.detector.sensitivityCurve[1], lw=2.5, label='_nolegend_')
                plt.scatter(peakFrequency_sw_shellThickness, peakAmplitude_sw_soundShell, c=T, cmap='coolwarm', marker='o',
                    s=49)
                plt.scatter(peakFrequency_sw_shellThickness_2, peakAmplitude_sw_soundShell_2, c=T, cmap='coolwarm',
                    marker='.', s=25)
                plt.scatter(peakFrequency_turb, peakAmplitude_turb, c=T, cmap='coolwarm', marker='x', s=49)
                plt.scatter(peakFrequency_turb_2, peakAmplitude_turb_2, c=T, cmap='coolwarm', marker='s', s=25)
                plt.xscale('log')
                plt.yscale('log')
                plt.xlim(left=min(peakFrequency_sw_shellThickness[0], peakFrequency_turb[0],
                    peakFrequency_sw_shellThickness_2[0], peakFrequency_turb_2[0]), right=max(0.1,
                    peakFrequency_sw_shellThickness[-1], peakFrequency_turb[-1], peakFrequency_sw_shellThickness_2[0],
                    peakFrequency_turb_2[0]))
                plt.ylim(top=100)
                plt.xlabel('$f_{\\mathrm{peak}} \\;\\; \\mathrm{[Hz]}$', fontsize=40)
                plt.ylabel('$h^2 \\Omega_{\\mathrm{peak}}$', fontsize=40)
                colorbar = plt.colorbar()
                colorbar.set_label(label='$T_* \\;\\; \\mathrm{[GeV]}$', fontsize=28)
                colorbar.ax.tick_params(labelsize=16)

                # See https://stackoverflow.com/a/47392973
                soundWaves = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10,
                    label='$\\mathrm{sound \\; waves \\; (separation)}$')
                soundWaves_2 = mlines.Line2D([], [], color='black', marker='.', linestyle='None', markersize=10,
                    label='$\\mathrm{sound \\; waves \\; (radius)}$')
                turbulence = mlines.Line2D([], [], color='black', marker='x', linestyle='None', markersize=10,
                    label='$\\mathrm{turbulence \\; (separation)}$')
                turbulence_2 = mlines.Line2D([], [], color='black', marker='s', linestyle='None', markersize=10,
                    label='$\\mathrm{turbulence \\; (radius)}$')
                plt.legend(handles=[soundWaves, soundWaves_2, turbulence, turbulence_2], fontsize=28, loc='lower left',
                    handletextpad=0.1)

            #ax = plt.gca()
            #leg = ax.get_legend()
            #leg.legend_handles[0].set_marker('o')
            #leg.legend_handles[0].set_color('black')
            #leg.legend_handles[1].set_color('black')
            finalisePlot('GW_peak_scatter')

            if not bCombined:
                plt.figure(figsize=(12, 8))
                plt.plot(T, SNR_regular, lw=2.5)
                plt.plot(T, SNR_soundShell, lw=2.5)
                plt.yscale('log')
                plotMilestoneTemperatures()
                plt.xlabel('$T_* \\;\\; \\mathrm{[GeV]}$', fontsize=40)
                plt.ylabel('$\\mathrm{SNR}$', fontsize=40)
                plt.legend(['$\\mathrm{lattice}$', '$\\mathrm{sound \\; shell}$'], fontsize=28)
                finalisePlot('SNR_vs_T')
            else:
                plt.figure(figsize=(12, 8))
                plt.plot(T, SNR_regular, lw=2.5)
                plt.plot(T, SNR_soundShell, lw=2.5)
                plt.plot(T, SNR_regular_2, lw=2.5, ls='--', label='_nolegend_', c=cycle[0])
                plt.plot(T, SNR_soundShell_2, lw=2.5, ls='--', label='_nolegend_', c=cycle[1])
                plt.yscale('log')
                plotMilestoneTemperatures()
                plt.xlabel('$T_* \\;\\; \\mathrm{[GeV]}$', fontsize=40)
                plt.ylabel('$\\mathrm{SNR}$', fontsize=40)
                plt.legend(['$\\mathrm{lattice}$', '$\\mathrm{sound \\; shell}$'], fontsize=28, loc='lower left')
                finalisePlot('SNR_vs_T')

            """plt.figure(figsize=(12, 8))
            plt.plot(T, [peakAmplitude_sw_soundShell[i] / peakAmplitude_sw_regular[i] for i in range(len(T))], lw=2.5)
            plotMilestoneTemperatures()
            plt.xlabel('$T \\;\\; \\mathrm{[GeV]}$', fontsize=40)
            plt.ylabel('$\\Omega_{\\mathrm{peak}}^{\\mathrm{sw}} \\mathrm{\\;\\; sound \\; shell \\; / \\; regular}$',
                fontsize=40)
            finalisePlot()"""

            if not bCombined:
                plt.figure(figsize=(12, 8))
                plt.plot(T, peakAmplitude_sw_regular, lw=2.5)
                plt.plot(T, peakAmplitude_sw_soundShell, lw=2.5)
                plt.plot(T, peakAmplitude_turb, lw=2.5)
                plt.yscale('log')
                plotMilestoneTemperatures()
                plt.xlabel('$T_* \\;\\; \\mathrm{[GeV]}$', fontsize=40)
                plt.ylabel('$h^2 \\Omega_{\\mathrm{peak}}$', fontsize=40)
                plt.legend(['$\\mathrm{lattice}$', '$\\mathrm{sound \\; shell}$', '$\\mathrm{turbulence}$'], fontsize=28)
                finalisePlot('Omega_peak_vs_T')
            else:
                plt.figure(figsize=(12, 8))
                plt.plot(T, peakAmplitude_sw_regular, lw=2.5)
                plt.plot(T, peakAmplitude_sw_soundShell, lw=2.5)
                plt.plot(T, peakAmplitude_turb, lw=2.5)
                plt.plot(T, peakAmplitude_sw_regular_2, lw=2.5, ls='--', label='_nolegend_', c=cycle[0])
                plt.plot(T, peakAmplitude_sw_soundShell_2, lw=2.5, ls='--', label='_nolegend_', c=cycle[1])
                plt.plot(T, peakAmplitude_turb_2, lw=2.5, ls='--', label='_nolegend_', c=cycle[2])
                plt.yscale('log')
                plotMilestoneTemperatures()
                plt.xlabel('$T_* \\;\\; \\mathrm{[GeV]}$', fontsize=40)
                plt.ylabel('$h^2 \\Omega_{\\mathrm{peak}}$', fontsize=40)
                plt.legend(['$\\mathrm{lattice}$', '$\\mathrm{sound \\; shell}$', '$\\mathrm{turbulence}$'], fontsize=28,
                    loc='upper left')
                finalisePlot('Omega_peak_vs_T')

            """plt.figure(figsize=(12, 8))
            plt.plot(T, kappa, lw=2.5)
            plt.plot(T, kappa_original, lw=2.5)
            plotMilestoneTemperatures()
            plt.xlabel('$T \\;\\; \\mathrm{[GeV]}$', fontsize=40)
            plt.ylabel('$\\kappa$', fontsize=40)
            plt.ylim(bottom=0)
            finalisePlot('kappa_sw_vs_T')"""

            plt.figure(figsize=(12, 8))
            plt.plot(T, K, lw=2.5)
            plotMilestoneTemperatures()
            plt.xlabel('$T_* \\;\\; \\mathrm{[GeV]}$', fontsize=40)
            plt.ylabel('$K$', fontsize=40)
            plt.ylim(bottom=0)
            #plt.legend(['$\\rho_\\mathrm{kin}/(\\rho_f + \\rho_\\mathrm{kin})$', '$\\rho_\\mathrm{kin}/\\rho_f$'],
            #    fontsize=28)
            finalisePlot('K_vs_T')

            """plt.figure(figsize=(12, 8))
            plt.plot(T, kappa, lw=2.5)
            plotMilestoneTemperatures()
            plt.xlabel('$T \\;\\; \\mathrm{[GeV]}$', fontsize=40)
            plt.ylabel('$\\kappa$', fontsize=40)
            plt.ylim(bottom=0)
            finalisePlot()

            plt.figure(figsize=(12, 8))
            plt.plot(T, alpha, lw=2.5)
            plotMilestoneTemperatures()
            plt.xlabel('$T \\;\\; \\mathrm{[GeV]}$', fontsize=40)
            plt.ylabel('$\\alpha$', fontsize=40)
            plt.ylim(bottom=0)
            finalisePlot()"""

            if not bCombined:
                plt.figure(figsize=(12, 8))
                plt.plot(T, peakFrequency_sw_bubbleSeparation, lw=2.5)
                plt.plot(T, peakFrequency_sw_shellThickness, lw=2.5)
                plt.plot(T, peakFrequency_turb, lw=2.5)
                plt.yscale('log')
                plotMilestoneTemperatures()
                plt.xlabel('$T_* \\;\\; \\mathrm{[GeV]}$', fontsize=40)
                plt.ylabel('$f_{\\mathrm{peak}} \\;\\; \\mathrm{[Hz]}$', fontsize=40)
                plt.legend(['$\\mathrm{lattice}$', '$\\mathrm{sound \\; shell}$', '$\\mathrm{turbulence}$'],
                    fontsize=28)
                finalisePlot('f_peak_vs_T')
            else:
                plt.figure(figsize=(12, 8))
                plt.plot(T, peakFrequency_sw_bubbleSeparation, lw=2.5)
                plt.plot(T, peakFrequency_sw_shellThickness, lw=2.5)
                plt.plot(T, peakFrequency_turb, lw=2.5)
                plt.plot(T, peakFrequency_sw_bubbleSeparation_2, lw=2.5, ls='--', label='_nolegend_', c=cycle[0])
                plt.plot(T, peakFrequency_sw_shellThickness_2, lw=2.5, ls='--', label='_nolegend_', c=cycle[1])
                plt.plot(T, peakFrequency_turb_2, lw=2.5, ls='--', label='_nolegend_', c=cycle[2])
                plt.yscale('log')
                plotMilestoneTemperatures()
                plt.xlabel('$T_* \\;\\; \\mathrm{[GeV]}$', fontsize=40)
                plt.ylabel('$f_{\\mathrm{peak}} \\;\\; \\mathrm{[Hz]}$', fontsize=40)
                plt.legend(['$\\mathrm{lattice}$', '$\\mathrm{sound \\; shell}$', '$\\mathrm{turbulence}$'],
                    fontsize=28, loc='lower left')
                finalisePlot('f_peak_vs_T')

            plt.figure(figsize=(12, 8))
            plt.plot(T, bubble_length_scale, lw=2.5)
            plt.plot(T, lengthScale_shellThickness, lw=2.5)
            plt.plot(T, meanBubbleRadius, lw=2.5)
            plt.yscale('log')
            plt.plot(T, lengthScale_beta, lw=2.5)
            if betaV > 0.: plt.axhline(lengthScale_betaV, lw=2, ls='--')
            plotMilestoneTemperatures()
            plt.xlabel('$T_* \\;\\; \\mathrm{[GeV]}$', fontsize=40)
            plt.ylabel('$\\mathrm{Length \\;\\; scale} \\;\\; \\mathrm{[GeV]^{-1}}$', fontsize=40)
            #plt.legend(['bubble separation', 'shell thickness', 'bubble separation (beta)'] +
            #    ['bubble separation (betaV)'] if betaV > 0. else [], fontsize=20)
            plt.legend(['$\\mathrm{bubble \\; separation} \\; (n)$', '$\\mathrm{shell \\; thickness}$',
                '$\\mathrm{bubble \\; radius}$', '$\\mathrm{bubble \\; separation} \\; (\\beta)$'] +
                (['$\\mathrm{bubble \\; separation} \\; (\\beta_V)$'] if betaV > 0 else []), fontsize=28,
                loc='upper left')
            #plt.ylim(bottom=0, top=max(bubble_length_scale[-1], lengthScale_shellThickness[-1]))
            #plt.ylim(bottom=0)
            finalisePlot('length_scale_vs_T')

            """plt.figure(figsize=(12, 8))
            plt.plot(T, Treh, lw=2.5)
            plt.plot(T, TrehApprox, lw=2.5, ls='--')
            plt.plot(T, T, lw=2, ls=':')
            #plotMilestoneTemperatures()
            plt.xlabel('$T \\;\\; \\mathrm{[GeV]}$', fontsize=40)
            plt.ylabel('$T_{\\mathrm{reh}} \\;\\; \\mathrm{[GeV]}$', fontsize=40)
            plt.ylim(bottom=0)
            #plt.tick_params(size=8, labelsize=28)
            #plt.margins(0, 0)
            #plt.tight_layout()
            #pathlib.Path(str(pathlib.Path('output/plots'))).mkdir(parents=True, exist_ok=True)
            #plt.savefig("output/plots/Treh_vs_T_BP1.pdf")
            #plt.show()
            finalisePlot()

            plt.figure(figsize=(12, 8))
            plt.plot(T, fluidVelocity, lw=2.5)
            plotMilestoneTemperatures()
            plt.xlabel('$T \\;\\; \\mathrm{[GeV]}$', fontsize=40)
            plt.ylabel('$\\overline{U}_f$', fontsize=40)
            plt.ylim(bottom=0)
            finalisePlot()

            plt.figure(figsize=(12, 8))
            plt.plot(T, upsilon, lw=2.5)
            plotMilestoneTemperatures()
            plt.xlabel('$T \\;\\; \\mathrm{[GeV]}$', fontsize=40)
            plt.ylabel('$\\Upsilon$', fontsize=40)
            plt.ylim(0, 1)
            finalisePlot()

            plt.figure(figsize=(12, 8))
            plt.plot(T, adiabaticIndex, lw=2.5)
            plotMilestoneTemperatures()
            plt.xlabel('$T \\;\\; \\mathrm{[GeV]}$', fontsize=40)
            plt.ylabel('$\\Gamma$', fontsize=40)
            plt.ylim(bottom=0)
            finalisePlot()"""

            plt.figure(figsize=(12, 8))
            plt.plot(T, csfSq, lw=2.5)
            plt.plot(T, cstSq, lw=2.5)
            plt.axhline(1/3, lw=2, ls='--')
            plotMilestoneTemperatures()
            plt.xlabel('$T_* \\;\\; \\mathrm{[GeV]}$', fontsize=40)
            plt.ylabel('$c_s^2$', fontsize=40)
            plt.legend(['false', 'true'], fontsize=20)
            finalisePlot('csSq_vs_T')

            plt.figure(figsize=(12, 8))
            plt.plot(T, csf, lw=2.5)
            plt.plot(T, cst, lw=2.5)
            plt.axhline(1/np.sqrt(3), lw=2, ls='--')
            plotMilestoneTemperatures()
            plt.xlabel('$T_* \\;\\; \\mathrm{[GeV]}$', fontsize=40)
            plt.ylabel('$c_s$', fontsize=40)
            plt.legend(['false', 'true'], fontsize=20)
            finalisePlot('cs_vs_T')

            """plt.figure(figsize=(12, 8))
            plt.plot(T, ndof, lw=2.5)
            plotMilestoneTemperatures()
            plt.xlabel('$T \\;\\; \\mathrm{[GeV]}$', fontsize=40)
            plt.ylabel('$g_*$', fontsize=40)
            finalisePlot()

            plt.figure(figsize=(12, 8))
            plt.plot(T, redshiftAmp, lw=2.5)
            plotMilestoneTemperatures()
            plt.xlabel('$T \\;\\; \\mathrm{[GeV]}$', fontsize=40)
            plt.ylabel('$\\mathcal{R}_\\Omega$', fontsize=40)
            finalisePlot()

            plt.figure(figsize=(12, 8))
            plt.plot(T, redshiftFreq, lw=2.5)
            plotMilestoneTemperatures()
            plt.xlabel('$T \\;\\; \\mathrm{[GeV]}$', fontsize=40)
            plt.ylabel('$\\mathcal{R}_f$', fontsize=40)
            finalisePlot()"""

def scanGWsWithParam(detectorClass, potentialClass, outputFolder, bForceAllTransitionsRelevant=False):
    peakAmplitude_sw: List[float] = []
    peakFrequency_sw_bubbleSeparation: List[float] = []
    peakFrequency_sw_shellThickness: List[float] = []
    peakAmplitude_turb: List[float] = []
    peakFrequency_turb: List[float] = []
    SNR_noColl: List[float] = []
    SNR_coll: List[float] = []
    K: List[float] = []
    vw: List[float] = []
    Tp: List[float] = []
    Treh: List[float] = []
    bubble_length_scale: List[float] = []
    lengthScale_shellThickness: List[float] = []
    adiabaticIndex: List[float] = []
    fluidVelocity: List[float] = []
    upsilon: List[float] = []
    csf: List[float] = []
    cst: List[float] = []
    ndof: List[float] = []
    redshiftAmp: List[float] = []
    redshiftFreq: List[float] = []
    #betaV: float = transitionReport['betaV']
    #GammaMax: float = transitionReport['GammaMax']
    #lengthScale_betaV: float = (np.sqrt(2*np.pi)*GammaMax/betaV)**(-1/3)
    kappa: List[float] = []
    alpha: List[float] = []
    N: List[float] = []
    H: List[float] = []
    TGamma: List[float] = []

    startTime = time.perf_counter()

    for i in range(100):
        outputSubfolder = outputFolder+str(i)+'/'
        gwa = GWAnalyser(detectorClass, potentialClass, outputSubfolder,
            bForceAllTransitionsRelevant=bForceAllTransitionsRelevant)
        transitionReport = gwa.relevantTransitions[0]
        gws_noColl = AnalyseIndividualTransition(gwa.phaseStructure, transitionReport, gwa.potential,
                gwa.detector)
        gws_coll = AnalyseIndividualTransition(gwa.phaseStructure, transitionReport, gwa.potential,
            gwa.detector)
        settings_coll = GWAnalysisSettings()
        settings_coll.kappaColl = 1.
        try:
            gws_noColl.determineGWs(settings=None)
        except:
            print('Failed fluid for index:', i)
            continue
        try:
            gws_coll.determineGWs(settings=settings_coll)
        except:
            print('Failed coll for index:', i)
            continue

        if gws_noColl.K == 0:
            print('Failed to calculate K for index:', i)
            continue
        gwFunc_noColl = gws_noColl.get_gw_total_func(soundShell=True)
        gwFunc_coll = gws_coll.get_gw_total_func()
        #SNR_noColl = gws_noColl.SNR(gwFunc_noColl)
        #SNR_coll = gws_noColl.SNR(gwFunc_coll)

        N.append(transitionReport['N'])
        H.append(transitionReport['Hp'])
        TGamma.append(transitionReport['TGammaMax'])
        kappa.append(gwa.potential.getParameterPoint()[0])
        peakAmplitude_sw.append(gws_noColl.peakAmplitude_sw_soundShell)
        peakFrequency_sw_bubbleSeparation.append(gws_noColl.peakFrequency_sw_bubbleSeparation)
        peakFrequency_sw_shellThickness.append(gws_noColl.peakFrequency_sw_shellThickness)
        peakAmplitude_turb.append(gws_noColl.peakAmplitude_turb)
        peakFrequency_turb.append(gws_noColl.peakFrequency_turb)
        SNR_noColl.append(gws_noColl.SNR(gwFunc_noColl))
        SNR_coll.append(gws_coll.SNR(gwFunc_coll))
        K.append(gws_noColl.K)
        vw.append(gws_noColl.vw)
        Tp.append(gws_noColl.T)
        Treh.append(gws_noColl.Treh)
        bubble_length_scale.append(gws_noColl.bubble_length_scale)
        lengthScale_shellThickness.append(gws_noColl.lengthScale_shellThickness)
        adiabaticIndex.append(gws_noColl.adiabaticIndex)
        fluidVelocity.append(gws_noColl.fluidVelocity)
        upsilon.append(gws_noColl.upsilon)
        csf.append(np.sqrt(gws_noColl.hydroVars.soundSpeedSqFalse))
        cst.append(np.sqrt(gws_noColl.hydroVars.soundSpeedSqTrue))
        ndof.append(gws_noColl.ndof)
        redshiftAmp.append(gws_noColl.redshiftAmp)
        redshiftFreq.append(gws_noColl.redshiftFreq)
        alpha.append(gws_noColl.alpha)

    print('Analysis took:', time.perf_counter() - startTime, 'seconds')
    print('kappa low:', kappa[0])
    print('kappa high:', kappa[-1])

    def finalisePlot():
        plt.tick_params(size=8, labelsize=18)
        plt.margins(0, 0)
        plt.show()

    plt.rcParams["text.usetex"] = True

    HR = [H[i]*bubble_length_scale[i] for i in range(len(Tp))]
    plt.figure(figsize=(12, 8))
    plt.scatter(Tp, HR, lw=2.5)
    plt.xlabel('$T_p$', fontsize=24)
    plt.ylabel('$H_* R_*$', fontsize=24)
    # plt.ylim(bottom=0.8, top=1.)
    plt.margins(0, 0)
    finalisePlot()

    Nthird = [n**(1/3) for n in N]
    invHRapprox = [TGamma[i]*Tp[i]*Nthird[i]/Treh[i]**2*(bubble_length_scale[i]*H[i]) for i in range(len(Tp))]

    plt.figure(figsize=(12, 8))
    plt.scatter(Tp, invHRapprox, lw=2.5)
    plt.xlabel('$T_p$', fontsize=24)
    plt.ylabel('$H_* R_* \\; \\mathrm{factor}$', fontsize=24)
    #plt.ylim(bottom=0.8, top=1.)
    plt.margins(0, 0)
    finalisePlot()

    plt.figure(figsize=(12, 8))
    plt.scatter(Tp, Nthird, lw=2.5)
    plt.xlabel('$T_p$', fontsize=24)
    plt.ylabel('$N(0)^{1/3}$', fontsize=24)
    plt.ylim(bottom=0)
    plt.margins(0, 0)
    finalisePlot()

    plt.figure(figsize=(12, 8))
    plt.scatter(Tp, vw, lw=2.5)
    plt.xlabel('$T_p$', fontsize=24)
    plt.ylabel('$v_w$', fontsize=24)
    finalisePlot()

    #indices = range(len(peakFreq_sw_bubbleSeparation))
    plt.figure(figsize=(12, 8))
    plt.plot(gwa.detector.sensitivityCurve[0], gwa.detector.sensitivityCurve[1], lw=2.5)
    plt.scatter(peakFrequency_sw_bubbleSeparation, peakAmplitude_sw, c=Tp, marker='o')
    plt.scatter(peakFrequency_turb, peakAmplitude_turb, c=Tp, marker='x')
    plt.xscale('log')
    plt.yscale('log')
    #plotMilestoneTemperatures()
    plt.xlabel('$f_{\\mathrm{peak}}$', fontsize=24)
    plt.ylabel('$\\Omega_{\\mathrm{peak}}$', fontsize=24)
    plt.legend(['noise', 'sw (regular)', 'sw (sound shell)', 'turb'], fontsize=20)
    colorbar = plt.colorbar()
    colorbar.set_label(label='$T$', fontsize=20)
    colorbar.ax.tick_params(labelsize=16)
    for i in range(1, len(plt.gca().get_legend().legendHandles)):
        plt.gca().get_legend().legendHandles[i].set_color('k')
        #handle.set_markeredgecolor('k')
    finalisePlot()

    plt.figure(figsize=(12, 8))
    plt.scatter(Tp, SNR_noColl, lw=2.5)
    plt.scatter(Tp, SNR_coll, lw=2.5)
    plt.xlabel('$T$', fontsize=24)
    plt.ylabel('$\\mathrm{SNR}$', fontsize=24)
    plt.legend(['fluid', 'coll'], fontsize=20)
    plt.ylim(bottom=0)#, top=max(SNR_noColl[-1], SNR_coll[-1]))
    finalisePlot()

    plt.figure(figsize=(12, 8))
    plt.scatter(Tp, peakAmplitude_sw, lw=2.5)
    plt.scatter(Tp, peakAmplitude_turb, lw=2.5)
    plt.xlabel('$T$', fontsize=24)
    plt.ylabel('$\\Omega_{\\mathrm{peak}}^{\\mathrm{sw}}$', fontsize=24)
    plt.ylim(bottom=0)#, top=max(peakAmplitude_sw_regular[-1], peakAmplitude_sw_soundShell[-1]))
    plt.legend(['sound', 'turb'], fontsize=20)
    finalisePlot()

    plt.figure(figsize=(12, 8))
    plt.scatter(Tp, K, lw=2.5)
    plt.xlabel('$T$', fontsize=24)
    plt.ylabel('$K$', fontsize=24)
    finalisePlot()

    plt.figure(figsize=(12, 8))
    plt.plot(Tp, peakFrequency_sw_bubbleSeparation, lw=2.5, marker='.')
    plt.plot(Tp, peakFrequency_sw_shellThickness, lw=2.5, marker='.')
    plt.xlabel('$T_p$', fontsize=24)
    plt.ylabel('$f_{\\mathrm{peak}}^{\\mathrm{sw}}$', fontsize=24)
    plt.legend(['bubble separation', 'shell thickness'], fontsize=20)
    plt.ylim(bottom=0)
    finalisePlot()

    #x = [Tp[i]*Tp[i]/Treh[i] for i in range(len(Tp))]
    y = [Tp[i]/Treh[i]*Nthird[i] for i in range(len(Tp))]

    """plt.figure(figsize=(12, 8))
    plt.plot(x, peakFrequency_sw_bubbleSeparation, lw=2.5, marker='.')
    plt.plot(x, peakFrequency_sw_shellThickness, lw=2.5, marker='.')
    plt.xlabel('$T_p^2/T_\\mathrm{reh}$', fontsize=24)
    plt.ylabel('$f_{\\mathrm{peak}}^{\\mathrm{sw}}$', fontsize=24)
    plt.legend(['bubble separation', 'shell thickness'], fontsize=20)
    plt.ylim(bottom=0)
    finalisePlot()"""

    plt.figure(figsize=(12, 8))
    plt.loglog(y, peakFrequency_sw_bubbleSeparation, lw=2.5, marker='.')
    plt.loglog(y, peakFrequency_sw_shellThickness, lw=2.5, marker='.')
    plt.xlabel('$N(0)^{1/3} \\, T_p/T_\\mathrm{reh}$', fontsize=24)
    plt.ylabel('$f_{\\mathrm{peak}}^{\\mathrm{sw}}$', fontsize=24)
    plt.legend(['bubble separation', 'shell thickness'], fontsize=20)
    plt.ylim(bottom=0)
    finalisePlot()

    plt.figure(figsize=(12, 8))
    plt.plot(Tp, peakFrequency_turb, lw=2.5)
    plt.xlabel('$T$', fontsize=24)
    plt.ylabel('$f_{\\mathrm{peak}}^{\\mathrm{turb}}$', fontsize=24)
    plt.ylim(bottom=0)
    finalisePlot()

    plt.figure(figsize=(12, 8))
    plt.scatter(Tp, bubble_length_scale, lw=2.5)
    plt.scatter(Tp, lengthScale_shellThickness, lw=2.5)
    plt.xlabel('$T$', fontsize=24)
    plt.ylabel('$\\mathrm{Length \\;\\; scale}$', fontsize=24)
    plt.legend(['bubble separation', 'shell thickness'],
        fontsize=20)
    plt.ylim(bottom=0)#, top=max(bubble_length_scale[-1], lengthScale_shellThickness[-1]))
    finalisePlot()

    TpForTrehPlot = Tp[:]
    TpForTrehPlot[0] = 0
    TrehApprox = [Tp[i]*(1 + alpha[i])**(1/4) for i in range(len(Tp))]

    plt.figure(figsize=(12, 8))
    plt.plot(TpForTrehPlot, Treh, lw=2.5)
    #plt.plot(TpForTrehPlot, TrehApprox, lw=2.5, ls='--')
    plt.plot(TpForTrehPlot, Tp, lw=1.75, ls=':', c='k')
    #plotMilestoneTemperatures()
    plt.xlabel('$T_p \\, \\mathrm{[GeV]}$', fontsize=40)
    plt.ylabel('$T_{\\mathrm{reh}} \\, \\mathrm{[GeV]}$', fontsize=40)
    plt.xlim(left=0, right=Tp[-1])
    plt.ylim(bottom=0, top=Treh[-1])
    plt.tick_params(size=8, labelsize=28)
    plt.margins(0, 0)
    plt.tight_layout()
    plt.margins(0, 0)
    pathlib.Path(str(pathlib.Path('output/nanograv'))).mkdir(parents=True, exist_ok=True)
    plt.savefig("output/nanograv/Treh_vs_Tp_scan.pdf")
    #finalisePlot()

    plt.figure(figsize=(12, 8))
    plt.scatter(Tp, fluidVelocity, lw=2.5)
    plt.xlabel('$T$', fontsize=24)
    plt.ylabel('$\\overline{U}_f$', fontsize=24)
    plt.ylim(bottom=0)
    finalisePlot()

    plt.figure(figsize=(12, 8))
    plt.scatter(Tp, upsilon, lw=2.5)
    plt.xlabel('$T$', fontsize=24)
    plt.ylabel('$\\Upsilon$', fontsize=24)
    plt.ylim(0, 1)
    finalisePlot()

    plt.figure(figsize=(12, 8))
    plt.scatter(Tp, adiabaticIndex, lw=2.5)
    plt.xlabel('$T$', fontsize=24)
    plt.ylabel('$\\Gamma$', fontsize=24)
    finalisePlot()

    plt.figure(figsize=(12, 8))
    plt.scatter(Tp, csf, lw=2.5)
    plt.scatter(Tp, cst, lw=2.5)
    plt.axhline(1/np.sqrt(3), lw=2, ls='--')
    plt.xlabel('$T$', fontsize=24)
    plt.ylabel('$c_s$', fontsize=24)
    plt.legend(['false', 'true'], fontsize=20)
    finalisePlot()

    plt.figure(figsize=(12, 8))
    plt.scatter(Tp, ndof, lw=2.5)
    plt.xlabel('$T$', fontsize=24)
    plt.ylabel('$g_*$', fontsize=24)
    finalisePlot()

    plt.figure(figsize=(12, 8))
    plt.scatter(Tp, redshiftAmp, lw=2.5)
    plt.xlabel('$T$', fontsize=24)
    plt.ylabel('$\\mathcal{R}_\\Omega$', fontsize=24)
    finalisePlot()

    plt.figure(figsize=(12, 8))
    plt.scatter(Tp, redshiftFreq, lw=2.5)
    plt.xlabel('$T$', fontsize=24)
    plt.ylabel('$\\mathcal{R}_f$', fontsize=24)
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

    if potentialClass == SMplusCubic:
        potential = potentialClass(*np.loadtxt(outputFolder + 'parameter_point.txt'), bUseBoltzmannSuppression=True)
    else:
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
        T = np.logspace(-3, np.log10(Tc*0.99), 100)

        for t in T:
            hydroVars = hydrodynamics.getHydroVars_new(fromPhase, toPhase, potential, t,
                phaseStructure.groundStateEnergyDensity)
            pf.append(hydroVars.pressureFalse)
            pt.append(hydroVars.pressureTrue)
            ef.append(hydroVars.energyDensityFalse)
            et.append(hydroVars.energyDensityTrue)
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
        plt.xscale('log')
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
def extractRelevantTransitions(report: dict, bForceAllTransitionsRelevant: bool = False) -> list[dict]:
    relevantTransitions = []

    try:
        isTransitionRelevant = [bForceAllTransitionsRelevant] * len(report['transitions'])

        if not bForceAllTransitionsRelevant:
            for transitionSequence in report['paths']:
                if transitionSequence['valid']:
                    for transitionID in transitionSequence['transitions']:
                        isTransitionRelevant[transitionID] = True

        for transition in report['transitions']:
            if isTransitionRelevant[transition['id']]:
                relevantTransitions.append(transition)

        return relevantTransitions
    except Exception:
        traceback.print_exc()
        return []


# This is for checking the difference between the old and new calculations of the mean bubble radius.
# The effect was < 0.5% for BP3.
def compareBubRad():
    with open('output/RSS/RSS_new_BP3/phase_history.json', 'r') as f:
        phr = json.load(f)
        tr1 = phr['transitions'][2]

    with open('output/RSS/RSS_new_BP3(old rad)/phase_history.json', 'r') as f:
        phr = json.load(f)
        tr2 = phr['transitions'][2]

    T1 = tr1['TSubsample']
    T2 = tr2['TSubsample']
    R1 = tr1['meanBubbleRadiusArray']
    R2 = tr2['meanBubbleRadiusArray']

    plt.plot(T1, R1)
    plt.plot(T2, R2)
    plt.yscale('log')
    plt.margins(0., 0.)
    plt.show()

def main(potentialClass, GWsOutputFolder, TSOutputFolder, detectorClass = LISA):
    gwa = GWAnalyser(detectorClass, potentialClass, TSOutputFolder, bForceAllTransitionsRelevant=False)
    # scan over reference temperature and make plots, as done in https://arxiv.org/abs/2309.05474
    gwa.scanGWs(GWsOutputFolder, bCombined=False)
    # Just run a single point at percolation temperature
    settings = GWAnalysisSettings()
    settings.bUseChapmanJouguetVelocity=True
    gwa.determineGWs(GWsOutputFolder, settings)
    # Use this for evaluating GWs using thermal params at the onset of percolation.
    #gwa.determineGWs_withColl()
    #scanGWsWithParam(detectorClass, potentialClass, outputFolder, bForceAllTransitionsRelevant=True)
    #hydroTester(potentialClass, outputFolder)

# when called as a script, read in arguments listed in the order
# 1 the model,
# 2 output directory,
# 3 input (ie the TS output),
# 4 and optionally the detector class, though only LISA is currently provided.
# if no arguments are provided it will run with the defaults provided below
if __name__ == "__main__":
    # new code
    default_model = RealScalarSingletModel_Boltz
    default_output_dir = 'GWsOutput/plots/'
    default_input_dir = 'output/RSS/RSS_BP3/'
    default_detector = LISA
    import sys

    print(sys.argv)
    print( "we have ", len(sys.argv), " arguments")
    # Check that the user has included enough parameters in the run command.
    if len(sys.argv) < 2:
        print('Since no arguments have been specifed running default model for the default GW detector, with default input folder and output folder.')
        print('If you wish to use a differemt model or input/output files or detector please either:')
        print('a) edit the default model, input and outfolder strings or detector set just below if __name__ == "__main__": in gws/gw_analyser.py')
        print('b) specify them at the command line as described in the README')
        main(default_model, default_output_dir, default_input_dir, default_detector)
        sys.exit(0)
    # read in model labels in lower case regardless of input case
    modelLabel = sys.argv[1].lower()
    print("modellabel set to ", modelLabel)
    modelLabels = ['toy', 'rss', 'rss_ht', 'smpluscubic']
    # The AnalysablePotential subclass corresponding to a particular model label.
    models = [ToyModel, RealScalarSingletModel_Boltz, RealScalarSingletModel_HT, SMplusCubic]
    # PhaseTracer script to run, specific to a particular model label.
    PT_scripts = ['run_ToyModel', 'run_RSS', 'run_RSS', 'run_supercool']
    # Extra arguments to pass to PhaseTracer, specific to a particular model label.
    PT_paramArrays = [[], ['-boltz'], ['-ht'], ['-boltz']]
    _potentialClass = None
    _PT_script = ''
    _PT_params = []

    # Attempt to match the input model label to the supported model labels.
    for i in range(len(models)):
        if modelLabel == modelLabels[i]:
            _potentialClass = models[i]
            _PT_script = PT_scripts[i]
            _PT_params = PT_paramArrays[i]
            break

    if _potentialClass is None:
        print(f'Invalid model label: {modelLabel}. Valid model labels are: {modelLabels}')
        sys.exit(1)
    # location for to utput GWs results.
    output_dir = sys.argv[2]
    print("output_dir set to ", output_dir)
    # location for TS output to be used to as input here to compute GWs
    TS_output_dir = sys.argv[3]
    if len(sys.argv) > 4:
        Detector = sys.argv[4]
    else:
        Detector = LISA
    main(RealScalarSingletModel_Boltz, output_dir, TS_output_dir, Detector)

# TODO: decide if we really need to do this here also, after I added GWs to command line interface
#      develop an interface to run main part of TransitionSolver
#       and then GWs and the same time.  Should try to avoid writing to json
#      file and then reading it in.
#      Inreface can work something like this:
#      When called as a script, read in argument for
#           # 1 the model,
#           # 2 output directory,
#           # 3 True if you want to run TransitionSolver from the beginning
#           #        for an input file specifying the parameters OR
#           #   False if you want run the gws module on results that have
#           #         already been obtained from running basic functinality
#           #         of TransitionSolver.
#           # 4 input: if 3 is True this will be the name of a file
#           #             specifying the input parameters of the model selected
#           #             in 1.
#           #          If 3 is False thsi will be the directory where output
#           #             from runinng TS (without GWs) is located 
#           #  5 optionally set the detector (default is LISA)
#       Could works something like this:
#       if __name__ == "__main__":                                                    
#         # new code                                                                  
#         default_output_dir = 'GWsOutput/plots/'                                     
#         default_input_dir = 'output/RSS/RSS_BP3/'                                   
#         default_model = RealScalarSingletModel_Boltz                                
#         default_detector = LISA                                                     
#         import sys                                                                  
#                                                                                     
#         print(sys.argv)                                                             
#         print( "we have ", len(sys.argv), " arguments")                             
#         # Check that the user has included enough parameters in the run command.    
#        if len(sys.argv) < 2:
#                print('Since no arguments have been specifed running default model for the default GW detector, with default input folder and output folder.')
#                print('If you wish to use a differemt model or input/output files or detector please either:')
#                print('a) edit the default model, input and outfolder strings or detector set just below if __name__ == "__main__": in gws/gw_analyser.py')
#                print('b) specify them at the command line as described in the README')
#                main(default_model, default_output_dir, default_input_dir, default_detector)
#                sys.exit(0)
#         # read in model labels in lower case regardless of input case
#         modelLabel = sys.argv[1].lower()
#         print("modellabel set to ", modelLabel)
#         modelLabels = ['toy', 'rss', 'rss_ht', 'smpluscubic']
#         # The AnalysablePotential subclass corresponding to a particular model label.
#         models = [ToyModel, RealScalarSingletModel_Boltz, RealScalarSingletModel_HT, SMplusCubic]
#         # PhaseTracer script to run, specific to a particular model label.
#         PT_scripts = ['run_ToyModel', 'run_RSS', 'run_RSS', 'run_supercool']
#         # Extra arguments to pass to PhaseTracer, specific to a particular model label.
#         PT_paramArrays = [[], ['-boltz'], ['-ht'], ['-boltz']]
#         _potentialClass = None
#          _PT_script = ''
#          _PT_params = []
#      
#          # Attempt to match the input model label to the supported model labels.
#          for i in range(len(models)):
#              if modelLabel == modelLabels[i]:
#                  _potentialClass = models[i]
#                  _PT_script = PT_scripts[i]
#                  _PT_params = PT_paramArrays[i]
#                  break
#      
#          if _potentialClass is None:
#              print(f'Invalid model label: {modelLabel}. Valid model labels are: {modelLabels}')
#              sys.exit(1)
#      
#          output_dir = sys.argv[2]
#          print("output_dir set to ", output_dir)
#          # boolean value indicat=ing whether we run TS from scratch or only run GWs for pre-exiting output
#          run_TS_and_GWs =  sys.argv[3]
#          # location for TS output to be used to as input here to compute GWs
#          #inputs = sys.argv[3]
#          if(run_TS_and_GWs == False):
#              input_file = sys.argv[4]
#          else:    
#              TS_output_dir = sys.argv[4]
#      # if an optional argument for detector is provided set it
#          if len(sys.argv) > 5:
#              Detector = sys.argv[5]
#          else:
#              # otherwise use default_detector (initially set to LISA but can be manually changed)
#              Detector = default_detector
#          #PA: I think for  scanGWs this is then enough and can call main - need to do that and then figure out calling with parameters after testing that works at least.
#          if len(sys.argv) < 6:
#              main(RealScalarSingletModel_Boltz, output_dir, TS_output_dir, Detector)      
#      
#      
