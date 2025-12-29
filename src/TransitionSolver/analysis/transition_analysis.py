"""
Transiton analysis
=====================
"""

from __future__ import annotations

import logging
import time
import json
import traceback
import sys
import warnings
from typing import Optional, Union
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from scipy.interpolate import lagrange

try:
    from cosmoTransitions.tunneling1D import ThinWallError
except:
    class ThinWallError(Exception):
        pass

from cosmoTransitions import pathDeformation

from ..gws import hydrodynamics
from ..gws.hydrodynamics import HydroVars
from ..util import integration
from ..util.events import notifyHandler
from ..models.analysable_potential import AnalysablePotential
from .phase_structure import Phase, Transition
from .. import geff

logger = logging.getLogger(__name__)

totalActionEvaluations = 0

# TODO: should move to a constants file.
GRAV_CONST = 6.7088e-39


class ActionSample:
    """
    Container for action sample data
    """
    def __init__(self, T=None, S3=None):
        self.T = T
        self.S3 = S3
        
    @property
    def SonT(self):
        return None if not self.is_valid else self.S3 / self.T
    
    @property
    def is_valid(self):
        return self.S3 is not None and self.T is not None and self.S3 > 0. and self.T > 0.

    def copy(self):
        return ActionSample(self.T, self.S3)

    def __str__(self):
        return f'(T: {self.T}, S/T: {self.SonT})'

    def __repr__(self):
        return str(self)


class ActionSampler:
    transitionAnalyser: 'TransitionAnalyser'

    # If this is true, CosmoTransitions will output details of the bounce action calculation, such as how many
    # iterations we used in the path deformation.
    bVerbose: bool = False
    # Whether a phase with a field value very close to zero compared to its characteristic value should be forced to
    # zero after optimisation, in preparation for action evaluation. This may reduce the effect of numerical errors
    # during optimisation. E.g. the field configuration for a phase might have phi1 = 1e-5, whereas it should be
    # identically zero. The phase is then shifted to phi1 = 0 before the action is evaluated. See
    # PhaseHistoryAnalysis.evaluateAction for details.
    bForcePhaseOnAxis: bool = False
    maxIter: int = 20
    numSplineSamples: int = 100
    # Make these smaller to increase the precision of CosmoTransitions' bounce action calculation.
    phitol: float = 1e-8
    xtol: float = 1e-8

    # precomputedT and precomputedSonT are lists of values for the S/T curve. These can be used to avoid sampling the
    # action curve explicitly, until the samples run out.
    def __init__(self, transitionAnalyser: 'TransitionAnalyser', minSonTThreshold: float, maxSonTThreshold: float,
            toleranceSonT: float, stepSizeMax=0.95, precomputedT: Optional[list[float]] = None, precomputedSonT:
            Optional[list[float]] = None):
        self.transitionAnalyser = transitionAnalyser

        # Copy properties for concision.
        self.bDebug = transitionAnalyser.bDebug
        self.potential = transitionAnalyser.potential
        self.fromPhase = transitionAnalyser.fromPhase
        self.toPhase = transitionAnalyser.toPhase
        self.Tmin = transitionAnalyser.Tmin
        self.Tmax = transitionAnalyser.Tmax

        self.precomputedT = precomputedT if precomputedT is not None else []
        self.precomputedSonT = precomputedSonT if precomputedSonT is not None else []

        if len(self.precomputedT) != len(self.precomputedSonT):
            self.precomputedT = []
            self.precomputedSonT = []
            self.bUsePrecomputedSamples = False
        else:
            self.bUsePrecomputedSamples = len(self.precomputedT) > 0

        self.T = []
        self.SonT = []

        self.subT = []
        self.subSonT = []
        self.subRhof = []
        self.subRhot = []

        self.lowerSonTData = []

        self.stepSize = -1
        self.stepSizeMax = stepSizeMax

        self.minSonTThreshold = minSonTThreshold
        self.maxSonTThreshold = maxSonTThreshold
        self.toleranceSonT = toleranceSonT

        notifyHandler.handleEvent(self, 'on_create')

    def calculateStepSize(self, low=None, mid=None, high=None):
        lowT = self.T[-1] if low is None else low.T
        midT = self.T[-2] if mid is None else mid.T
        highT = self.T[-3] if high is None else high.T
        lowSonT = self.SonT[-1] if low is None else low.SonT
        midSonT = self.SonT[-2] if mid is None else mid.SonT
        highSonT = self.SonT[-3] if high is None else high.SonT

        # Check for linearity. Create a line between the lowest and highest temperature points, and check how far from
        # the average value the middle temperature point is.
        interpValue = lowSonT + (highSonT - lowSonT) * (midT - lowT) / (highT - lowT)
        linearity = 1 - 2*abs(midSonT - interpValue) / abs(highSonT - lowSonT)

        if self.stepSize == -1:
            self.stepSize = min(self.stepSizeMax, 1 - (abs(midSonT - lowSonT) / midSonT))

        # This gives (lin, stepFactor): (0.99, 1.0315), (0.98, 0.9186), (0.94, 0.8) and 0.5 for lin <= 0.8855.
        stepFactor = max(0.5, 0.8 + 0.4*((linearity - 0.94)/(1 - 0.94))**3)
        self.stepSize = min(self.stepSizeMax, 1 - stepFactor*(1 - self.stepSize))

        logging.debug(f'{self.stepSize=}')
        logging.debug(f'{stepFactor=}')
        logging.debug(f'{linearity=}')

    def getNextSample(self, sampleData: ActionSample, Gamma: list[float], numBubbles: float, Tmin: float)\
            -> (bool, str):
        # If we are already near T=0, the transition is assumed to not progress from here. We consider only bubble
        # nucleation via thermal fluctuations.
        #if len(self.T) > 0 and sampleData.T <= 0.001:
        #    return False, 'Freeze out'

        # If we are already near the minimum temperature allowed for phase transitions in this potential, we assume the
        # transition will not progress from here. Or, if it does, we cannot accurately determine its progress due to
        # external effects (like other cosmological events) that we don't handle.
        if len(self.T) > 0 and sampleData.T <= self.potential.minimum_temperature:
            return False, 'Reached minimum temperature'

        # Remove all stored data points whose temperature is larger than the last sampled temperature.
        while len(self.lowerSonTData) > 0 and self.lowerSonTData[-1].T >= self.T[-1]:
            self.lowerSonTData.pop()

        if self.bUsePrecomputedSamples:
            if len(self.T) < len(self.precomputedT):
                self.T.append(self.precomputedT[len(self.T)])
                self.SonT.append(self.precomputedSonT[len(self.SonT)])
                sampleData.T = self.T[-1]
                sampleData.SonT = self.SonT[-1]
                return True, 'Precomputed'
            elif len(self.T) == len(self.precomputedT):
                # TODO: not exactly sure if this is necessary, but probably is.
                self.calculateStepSize()

        # Construct a quadratic Lagrange interpolant from the three most recent action samples.
        quadInterp = lagrange(self.T[-3:], self.SonT[-3:])
        # Extrapolate with the same step size as between the last two samples.
        Tnew = max(self.Tmin*1.001, 2*self.T[-1] - self.T[-2])
        SonTnew = quadInterp(Tnew)

        # If we are sampling the same point because we've reached Tmin, then the transition cannot progress any
        # further.
        if self.T[-1] == self.Tmin*1.001:
            logger.debug('Already sampled near Tmin ={}. Transition analysis halted', sampleData.T)
            return False, 'Reached Tmin'

        # Determine the nucleation rate for nearby temperatures under the assumption that quadratic extrapolation is
        # appropriate.

        GammaNew = self.transitionAnalyser.calculateGamma(Tnew, SonTnew)
        GammaCur = self.transitionAnalyser.calculateGamma(self.T[-1], self.SonT[-1])
        GammaPrev = self.transitionAnalyser.calculateGamma(self.T[-2], self.SonT[-2])

        def nearMaxNucleation() -> bool:
            dSdTnew = (self.SonT[-1] - SonTnew)/(self.T[-1] - Tnew)
            dSdT = (self.SonT[-2] - self.SonT[-1])/(self.T[-2] - self.T[-1])
            # If the relative derivative is changing rapidly, then we are near the minimum.
            # TODO: given the exponential curve, might need to change this derivative test.
            derivTest = dSdTnew/dSdT < 0.8 if dSdT > 0 else dSdTnew/dSdT > 1.25
            return GammaNew > 0 and (GammaNew/GammaCur < 0.8*GammaCur/GammaPrev or derivTest)

        stepFactor = 1.
        nearMaxFactor = 0.7

        if numBubbles < 1e-4:
            if GammaNew < GammaCur:
                stepFactor = 2.
            else:
                if nearMaxNucleation():
                    stepFactor = nearMaxFactor
                else:
                    extraBubbles = self.calculateExtraBubbles(Tnew, SonTnew, Tmin)

                    if extraBubbles + numBubbles > 1e-4:
                        if extraBubbles > numBubbles*10:
                            if extraBubbles > numBubbles*100:
                                stepFactor = 0.5
                            else:
                                stepFactor = 0.75
                        elif extraBubbles < numBubbles:
                            stepFactor = 2.
                    elif extraBubbles + numBubbles < 1e-5:
                        stepFactor = 2.
                    else:
                        stepFactor = 1.5
        elif numBubbles < 0.1:
            if GammaNew < GammaCur:
                stepFactor = 1.4
            else:
                if nearMaxNucleation():
                    stepFactor = nearMaxFactor
                else:
                    extraBubbles = self.calculateExtraBubbles(Tnew, SonTnew, Tmin)

                    if extraBubbles > numBubbles*5:
                        if extraBubbles > numBubbles*25:
                            stepFactor = 0.5
                        else:
                            stepFactor = 0.75
                    elif extraBubbles < numBubbles:
                        stepFactor = 1.4
        elif numBubbles < 1:
            if GammaNew < GammaCur:
                stepFactor = 1.1
            else:
                if nearMaxNucleation():
                    stepFactor = nearMaxFactor
                else:
                    extraBubbles = self.calculateExtraBubbles(Tnew, SonTnew, Tmin)

                    if extraBubbles > 0.2*numBubbles:
                        if extraBubbles > 0.5*numBubbles:
                            stepFactor = 0.5
                        else:
                            stepFactor = 0.75
                    elif extraBubbles < 0.1*numBubbles:
                        stepFactor = 1.2
        else:  # numBubbles > 1
            if GammaNew < GammaCur:
                if GammaCur < GammaPrev:
                    if GammaNew < 1e-3*max(Gamma):
                        stepFactor = 1.5
                    else:
                        stepFactor = 1.2
                # Found minimum of Gamma, sample more densely.
                else:
                    stepFactor = 0.6
            else:
                if nearMaxNucleation():
                    stepFactor = nearMaxFactor
                else:
                    if numBubbles < 100:
                        extraBubbles = self.calculateExtraBubbles(Tnew, SonTnew, Tmin)

                        if numBubbles < 10:
                            if extraBubbles < numBubbles:
                                if extraBubbles < 0.5*numBubbles:
                                    stepFactor = 3.
                                else:
                                    stepFactor = 2.
                            elif extraBubbles < 2*numBubbles:
                                stepFactor = 2.
                        else:
                            stepFactor = 1.5

        #if numBubbles < 1:
        #    Tnew = max(0.001, self.T[-1] - 0.2)
        #else:
        # TODO: added for dense sampling for noNucleation-c4Scan-new.
        if numBubbles < 1 and SonTnew < 160:
            if stepFactor > 1:
                stepFactor = 1 + 0.85*(stepFactor - 1)
            else:
                stepFactor *= 0.85

        Tnew = max(self.Tmin*1.001, self.T[-1] - max(stepFactor*(self.T[-2] - self.T[-1]),
            self.potential.minimum_temperature))

        # Prevent large steps near T=Tmin causing large errors in the interpolated action from affecting ultracooled
        # transitions.
        if (Tnew - self.Tmin) < 0.5*(self.T[-1] - self.Tmin) and (self.T[-1] - self.Tmin) > 5.:
            Tnew = 0.5*(self.Tmin + self.T[-1])

        # Hack for better resolution in supercool GW plots.
        #maxStep = 0.1 # BP1 and BP2
        #maxStep = 0.3 # BP3 and BP4

        #if abs(Tnew - self.T[-1]) > maxStep:
        #    Tnew = self.T[-1] - maxStep

        sampleData.T = Tnew

        self.evaluateAction(sampleData)

        if not sampleData.is_valid:
            logger.info('Failed to evaluate action at trial temperature T = {}', sampleData.T)
            return False, 'Action failed'

        self.T.append(sampleData.T)
        self.SonT.append(sampleData.SonT)

        return True, 'Success'

    def calculateExtraBubbles(self, Tnew: float, SonTnew: float, Tmin: float) -> float:
        extraBubbles = 0
        numPoints = 20

        TList = np.linspace(self.T[-1], Tnew, numPoints)
        # TODO: replace with quadratic interpolation.
        SonTList = np.linspace(self.SonT[-1], SonTnew, numPoints)
        GammaList = self.transitionAnalyser.calculateGamma(TList, SonTList)
        energyStart = hydrodynamics.energy_density_from_phase(self.fromPhase, self.toPhase, self.potential,
            TList[0]) - self.transitionAnalyser.groud_state_energy_density
        energyEnd = hydrodynamics.energy_density_from_phase(self.fromPhase, self.toPhase, self.potential,
            TList[-1]) - self.transitionAnalyser.groud_state_energy_density
        # TODO: replace with quadratic interpolation.
        energyDensityList = np.linspace(energyStart, energyEnd, numPoints)
        HList = hubble_squared_from_energy_density(energyDensityList)
        integrandList = [GammaList[i]/(TList[i]*HList[i]**2) for i in range(numPoints)]

        for i in range(1, numPoints):
            extraBubbles += 0.5*(integrandList[i] + integrandList[i-1]) * (TList[i-1] - TList[i])

        return extraBubbles

    def getNumSubsamples(self, minSonTThreshold, maxSonTThreshold):
        # Interpolate between the last two sample points, creating a densely sampled line which we can integrate.
        # Increase the sampling density as S/T decreases as we need greater resolution for S/T where the nucleation
        # probability rises exponentially.
        quickFactor = 1.
        # return 3 #max(5, int(100*(self.T[-2] - self.T[-1])))
        #numSamples = int(quickFactor*(100 + 1000*))
        # TODO: make the number of interpolation samples have some physical motivation (e.g. ensure that the true
        #  vacuum fraction can change by no more than 0.1% of its current value across each sample).
        actionFactor = abs(self.SonT[-2]/self.SonT[-1] - 1) / (1 - minSonTThreshold/maxSonTThreshold)
        dTFactor = abs(min(2,self.T[-2]/self.T[-1]) - 1) / (1 - self.Tmin/self.T[0])
        numSamples = max(1, int(quickFactor*(2 + np.sqrt(1000*(actionFactor + dTFactor)))))
        logger.debug('Num samples: {}, {},g {}', numSamples, actionFactor, dTFactor)
        return numSamples

    # Stores newData in lowerSonTData, maintaining the sorted order.
    def storeLowerSonTData(self, newData):
        if len(self.lowerSonTData) == 0 or self.lowerSonTData[-1].T <= newData.T:
            self.lowerSonTData.append(newData)
        else:
            i = 0

            while i < len(self.lowerSonTData) and newData.T > self.lowerSonTData[i].T:
                i += 1

            self.lowerSonTData.insert(i, newData)

    # Throws ThinWallError from CosmoTransitions.
    # Throws unhandled exceptions from CosmoTransitions.
    # Throws unhandled InvalidTemperatureException from find_phase_at_t.
    def evaluateAction(self, data: ActionSample) -> tuple[np.ndarray, np.ndarray]:
        # Do optimisation.
        T = data.T
        fromFieldConfig = self.fromPhase.find_phase_at_t(T, self.potential)
        toFieldConfig = self.toPhase.find_phase_at_t(T, self.potential)

        fieldSeparationScale = 0.001*self.potential.get_field_scale()

        # If the phases merge together, reset them to their previous values. This can happen for subcritical transitions,
        # where there is virtually no potential barrier when a new phase appears.
        if np.linalg.norm(fromFieldConfig - toFieldConfig) < fieldSeparationScale:
            # TODO: find a solution to this problem if it shows up.
            raise Exception("The 'from' and 'to' phases merged after optimisation, in preparation for action evaluation.")

        if self.bForcePhaseOnAxis:
            for i in range(len(fromFieldConfig.shape)):
                if abs(fromFieldConfig[i]) < fieldSeparationScale:
                    fromFieldConfig[i] = 0.0
                if abs(toFieldConfig[i]) < fieldSeparationScale:
                    toFieldConfig[i] = 0.0

        return self.evaluateAction_supplied(data, fromFieldConfig, toFieldConfig)

    # Throws ThinWallError from CosmoTransitions.
    # Throws unhandled exceptions from CosmoTransitions.
    def evaluateAction_supplied(self, data: ActionSample, fromFieldConfig: np.ndarray, toFieldConfig: np.ndarray)\
            -> tuple[np.ndarray, np.ndarray]:
        tunneling_findProfile_params = {'phitol': self.phitol, 'xtol': self.xtol}

        T = data.T
        startTime = time.perf_counter()

        logger.debug('Evaluating action at T = {}', T)

        def V(X): return self.potential(X, T)
        def gradV(X): return self.potential.grad(X, T)
        
        global totalActionEvaluations
        totalActionEvaluations += 1

        with redirect_stdout(sys.stdout if self.bVerbose else None):
            action = pathDeformation.fullTunneling([toFieldConfig, fromFieldConfig], V, gradV,
                V_spline_samples=self.numSplineSamples, maxiter=self.maxIter,
                verbose=self.bVerbose, tunneling_findProfile_params=tunneling_findProfile_params).action


        # Update the data, including changes to the phases and temperature if the calculation previously failed.
        data.T = T
        data.S3 = action

        if data.is_valid:
            logger.debug(f'Successfully evaluated action S = {data.SonT} in {time.perf_counter() - startTime} seconds.')
        else:
            logger.debug(f'Obtained nonsensical action S = {data.SonT} in {time.perf_counter() - startTime} seconds.')

        return fromFieldConfig, toFieldConfig


class ActionCurveShapeAnalysisData:
    def __init__(self):
        self.desiredData = None
        self.nextBelowDesiredData = None
        self.storedLowerActionData = []
        self.actionSamples = []
        # TODO: does the default choice for this ever matter? Are there cases where we don't identify it but still
        #  continue the transition analysis? Maybe for fine-tuned subcritical transitions that nucleate quickly but
        #  don't complete quickly?
        self.bBarrierAtTmin = False
        self.error = False

    def copyDesiredData(self, sample):
        self.desiredData = sample.copy()

    def copyNextBelowDesiredData(self, sample):
        self.nextBelowDesiredData = sample.copy()

    def copyStoredLowerActionData(self, samples):
        for sample in samples:
            self.storedLowerActionData.append(sample.copy())

    def copyActionSamples(self, samples):
        for sample in samples:
            # samples is a list of tuples of primitive types, so no need for a manual deep copy.
            if sample[2]:  # is_valid
                self.actionSamples.append(sample)


class TransitionAnalyser:
    bDebug: bool = False
    bPlot: bool = False
    # Optimisation: check whether completion can occur before reaching T=0. If it cannot, stop the transition analysis.
    bCheckPossibleCompletion: bool = True
    # Whether transition analysis should continue after finding the completion temperature, all the way down to the
    # lowest temperature for which the transition is possible. If this is false, transition analysis stops as soon as
    # completion is found (default behaviour).
    bAnalyseTransitionPastCompletion: bool = False
    bAllowErrorsForTn: bool = True
    bReportAnalysis: bool = False
    timeout_phaseHistoryAnalysis: float = -1.
    bUseChapmanJouguetVelocity: float = False

    potential: AnalysablePotential
    transition: Transition
    fromPhase: Phase
    toPhase: Phase
    groud_state_energy_density: float

    # The lowest temperature for which the transition is still possible. We don't want to check below this, as one
    # of the phases may have disappeared, their stability may reverse, or T=0 may have been reached.
    Tmin: float = 0.
    Tmax: float = 0.
    Tstep: float = 0.

    actionSampler: ActionSampler

    bComputeSubsampledThermalParams: bool

    # TODO: make vw a function of this class that can be overriden. Currently it is obtained from the transition.
    def __init__(self, potential: AnalysablePotential, properties, fromPhase: Phase, toPhase: Phase,
            groud_state_energy_density: float, Tmin: float = 0., Tmax: float = 0.):
        self.potential = potential
        self.properties = properties
        self.fromPhase = fromPhase
        self.toPhase = toPhase
        self.groud_state_energy_density = groud_state_energy_density
        self.Tmin = Tmin
        self.Tmax = Tmax

        if self.Tmin == 0:
            # The minimum temperature for which both phases exist, and prevent analysis below the effective potential's
            # cutoff temperature. Below this cutoff temperature, external effects may dramatically affect the phase
            # transition and cannot be captured here in a generic way.
            self.Tmin = max(self.fromPhase.T[0], self.toPhase.T[0], self.potential.minimum_temperature)

        if self.Tmax == 0:
            # The transition is not evaluated subcritically.
            self.Tmax = self.properties.Tc

        self.Tstep = max(0.0005*min(self.fromPhase.T[-1], self.toPhase.T[-1]), 0.0001*self.potential.get_temperature_scale())

        self.bComputeSubsampledThermalParams = False

        notifyHandler.handleEvent(self, 'on_create')

    def getBubbleWallVelocity(self, hydrovars: HydroVars) -> float:
        if self.bUseChapmanJouguetVelocity:
            logging.debug(f'{hydrovars=}')
            vw = hydrovars.cj_velocity
            
            if np.isnan(vw) or vw > 1.:
                logger.warning("vw = {}. Adjusting to 1", vw)
                return 1.

            return vw

        return self.properties.vw

    # TODO: need to handle subcritical transitions better. Shouldn't use integration if the max sampled action is well
    #  below the nucleation threshold. Should treat the action as constant or linearise it and estimate transition
    #  temperatures under that approximation.
    def analyseTransition(self, startTime: float = -1.0, precomputedActionCurveFileName: str = ''):
        # TODO: this should depend on the scale of the transition, so make it configurable.
        # Estimate the maximum significant value of S/T by finding where the instantaneous nucleation rate multiplied by
        # the maximum possible duration of the transition is O(1). This is highly conservative, but intentionally so
        # because we only sample maxSonTThreshold within some (loose) tolerance.
        maxSonTThreshold = self.estimateMaximumSignificantSonT() + 80
        minSonTThreshold = 80.0
        toleranceSonT = 3.0

        precomputedT, precomputedSonT, bFinishedAnalysis = loadPrecomputedActionData(precomputedActionCurveFileName,
            self.properties, maxSonTThreshold)

        if bFinishedAnalysis:
            return

        self.actionSampler = ActionSampler(self, minSonTThreshold, maxSonTThreshold, toleranceSonT,
            precomputedT=precomputedT, precomputedSonT=precomputedSonT)

        logging.debug(f'{self.Tmin=}')
        logging.debug(f'{self.Tmax=}')
        logging.debug(f'{self.properties.Tc=}')
        logging.debug(f'{self.properties.vw=}')

        if not precomputedT:
            # TODO: we don't use allSamples anymore.
            sampleData, allSamples = self.primeTransitionAnalysis(startTime)

            if self.timeout_phaseHistoryAnalysis > 0:
                endTime = time.perf_counter()
                if endTime - startTime > self.timeout_phaseHistoryAnalysis:
                    return

            if sampleData is None:
                self.properties.bFoundNucleationWindow = False
                return

            self.properties.bFoundNucleationWindow = True

            # Remove any lowerSonTData points that are very close together. We don't need to sample the S/T curve extremely
            # densely (a spacing of 1 is more than reasonable), and doing so causes problems with the subsequent steps along
            # the curve TODO: to be fixed anyway!
            if self.actionSampler.lowerSonTData:
                keepIndices = [len(self.actionSampler.lowerSonTData)-1]
                for i in range(len(self.actionSampler.lowerSonTData)-2, -1, -1):
                    # Don't discard the point if it is separated in temperature from the almost degenerate S/T value already
                    # stored.
                    if abs(self.actionSampler.lowerSonTData[i].SonT -
                            self.actionSampler.lowerSonTData[keepIndices[-1]].SonT) > 1 or\
                            abs(self.actionSampler.lowerSonTData[i].T -
                            self.actionSampler.lowerSonTData[keepIndices[-1]].T) >\
                            self.potential.get_temperature_scale()*0.001:
                        keepIndices.append(i)
                    else:
                        logger.debug('Removing stored lower S/T data {} because it is too close to {}', self.actionSampler.lowerSonTData[i], self.actionSampler.lowerSonTData[keepIndices[-1]])

                self.actionSampler.lowerSonTData = [self.actionSampler.lowerSonTData[i] for i in keepIndices]
        else:
            self.properties.bFoundNucleationWindow = True

            sampleData = ActionSample()
            self.actionSampler.getNextSample(sampleData, [], 0., self.Tmin)
            self.actionSampler.getNextSample(sampleData, [], 0., self.Tmin)

        # We finally have a temperature where S/T is not much smaller than the value we started with (which was close to
        # maxSonTThreshold). From here we can use the separation in these temperatures and S/T values to predict reasonable
        # temperatures to efficiently sample S/T in the range [minSonTThreshold, maxSonTThreshold] until the nucleation
        # temperature is found.

        # If we continue to assume linearity, we will overestimate the rate at which minSonTThreshold is reached. Therefore,
        # we can take large steps in temperature and dynamically update the step size based on the new prediction of when
        # minSonTThreshold will be reached. Further, we can adjust the step size based on how well the prediction matches
        # the observed value. If the prediction is good, we can increase the step size as the S/T curve must be close to
        # linear in this region. If the prediction is bad, the S/T must be noticeably non-linear in this region and thus we
        # need a smaller step size for accurate sampling.

        numBubblesIntegrand = [0.]
        numBubblesCorrectedIntegrand = [0.]
        numBubblesIntegral = [0.]
        numBubblesCorrectedIntegral = [0.]
        Vext = [0.]
        Pf = [1.]
        Gamma = [0.]
        bubbleNumberDensity = [0.]
        meanBubbleRadiusArray = [0.]
        meanBubbleSeparationArray = [0.]
        # TODO: need an initial value...
        betaArray = [0.]
        hydroVars = [self.getHydroVars(self.actionSampler.T[0])]
        H = [hydroVars[0].hubble_constant]
        logger.debug("calling getBubbleWallVelocity in analyseTransition...")
        vw = [self.getBubbleWallVelocity(hydroVars[0])]

        radDensityPrefactor = np.pi**2/30
        fourPiOnThree = 4/3*np.pi

        # =================================================
        # Calculate the maximum temperature result.
        # This gives a boundary condition for integration.
        T4 = self.actionSampler.T[0]**4

        Gamma[0] = T4 * (self.actionSampler.SonT[0] / (2*np.pi))**(3/2) * np.exp(-self.actionSampler.SonT[0])

        numBubblesIntegrand[0] = Gamma[0] / (self.actionSampler.T[0]*H[0]**4)
        numBubblesCorrectedIntegrand[0] = numBubblesIntegrand[0]

        # The temperature for which S/T is minimised. This is important for determining the bubble number density. If this
        # minimum S/T value is encountered after the percolation temperature, then it can be ignored. It is only important
        # if it occurs before percolation.
        TAtSonTmin = 0

        Teq = 0

        bFirst = True

        # When the fraction of space remaining in the false vacuum falls below this threshold, the transition is considered
        # to be complete.
        completionThreshold = 1e-2

        def outerFunction_trueVacVol(x):
            return Gamma[x] / (self.actionSampler.subT[x]**4 * H[x])

        def innerFunction_trueVacVol(x):
            return vw[x] / H[x]

        def outerFunction_avgBubRad(x):
            return Gamma[x]*Pf[x] / (self.actionSampler.subT[x]**4 * H[x])

        def innerFunction_avgBubRad(x):
            return vw[x] / H[x]

        def sampleTransformationFunction(x):
            return self.actionSampler.subT[x]

        # We need three (two) data points before we can initialise the integration helper for the true vacuum volume
        # (average bubble radius). We wait for four data points to be ready (see below for why), then initialise both at the
        # same time.
        integrationHelper_trueVacVol = None
        integrationHelper_avgBubRad = None

        Tn = Tnbar = Tp = Te = Tf = Ts1 = Ts2 = -1
        indexTp = indexTf = -1

        # Don't check for Teq if the transition is subcritical (or evaluated subcritically) and the vacuum energy density
        # already exceeds the radiation energy density at the maximum temperature.
        #bCheckForTeq = (not transition.subcritical and Tmax == transition.Tc) or\
        #    calculateEnergyDensityAtT(Tmax)[0] <= radDensity*Tmax**4

        #rho0 = hydrodynamics.energy_density_from_phase(self.fromPhase, self.toPhase, self.potential,
        #    self.actionSampler.T[0], forFromPhase=True) - self.groud_state_energy_density
        rho0 = hydroVars[0].energyDensityFalse
        Temp = self.actionSampler.T[0]
        phi = self.fromPhase.find_phase_at_t(Temp, self.potential)
        rhoR = radDensityPrefactor * geff.field_dependent_dof(self.potential, phi, Temp) * Temp**4

        bCheckForTeq = rho0 - 2*rhoR < 0

        if not bCheckForTeq:
            logger.debug('Not checking for Teq since rho_V > rho_R at Tmax.')

        physicalVolume = [3]

        percolationThreshold_Pf = 0.71
        percolationThreshold_Vext = -np.log(percolationThreshold_Pf)

        # The index in the simulation we're up to analysing. Note that we always sample 1 past this index so we can
        # interpolate between T[simIndex] and T[simIndex+1].
        simIndex = 0

        # Keep sampling until we have identified the end of the phase transition or that the transition doesn't complete.
        # If bCheckPossibleCompletion is false we determine it does not complete only if we get to T=0 and it has not,
        # otherewise we can also determine this when transitionCouldComplete returns false.
        while not self.bCheckPossibleCompletion or self.transitionCouldComplete(maxSonTThreshold + toleranceSonT, Pf):
            # If the action begins increasing with decreasing temperature.
            if TAtSonTmin == 0 and self.actionSampler.SonT[simIndex+1] > self.actionSampler.SonT[simIndex]:
                # TODO: do some form of interpolation.
                TAtSonTmin = self.actionSampler.T[simIndex]
                SonTmin = self.actionSampler.SonT[simIndex]
                self.properties.Tmin = TAtSonTmin
                self.properties.SonTmin = SonTmin

            numSamples = self.actionSampler.getNumSubsamples(minSonTThreshold, maxSonTThreshold)
            T = np.linspace(self.actionSampler.T[simIndex], self.actionSampler.T[simIndex+1], numSamples)
            # We can only use quadratic interpolation if we have at least 3 action samples, which occurs for simIndex > 0.
            if simIndex > 0:
                quadInterp = lagrange(self.actionSampler.T[simIndex-1:], self.actionSampler.SonT[simIndex-1:])
                SonT = quadInterp(T)
            else:
                SonT = np.linspace(self.actionSampler.SonT[simIndex], self.actionSampler.SonT[simIndex+1], numSamples)

            #rhof1, rhot1 = hydrodynamics.calculateEnergyDensityAtT(self.fromPhase, self.toPhase, self.potential,
            #    self.actionSampler.T[simIndex])
            #rhof2, rhot2 = hydrodynamics.calculateEnergyDensityAtT(self.fromPhase, self.toPhase, self.potential,
            #    self.actionSampler.T[simIndex+1])
            T1 = self.actionSampler.T[simIndex]
            T2 = self.actionSampler.T[simIndex+1]
            hydroVars1 = self.getHydroVars(T1)
            hydroVars2 = self.getHydroVars(T2)
            hydroVarsInterp = [hydrodynamics.interpolate_hydro_vars(hydroVars1, hydroVars2, t) for t in T]

            dT = T[0] - T[1]

            # Since we skip the first element of the subsamples in the following loop, make sure to add the very first
            # subsample of the integration to the corresponding lists. This only needs to be done for the first sampling
            # iteration, as the next iteration's first subsample is this iteration's last subsample (i.e. it will
            # already have been added).
            if bFirst:
                self.actionSampler.subT.append(T[0])
                self.actionSampler.subSonT.append(SonT[0])
                self.actionSampler.subRhof.append(hydroVarsInterp[0].energyDensityFalse)
                self.actionSampler.subRhot.append(hydroVarsInterp[0].energyDensityTrue)
                bFirst = False

            # ==========================================================================================================
            # Begin subsampling.
            # ==========================================================================================================

            # Don't handle the first element of the list, as it is either the boundary condition for the integration (if
            # this is the first integrated data point), or was handled as the last element of the previous data point.
            for i in range(1, len(T)):
                self.actionSampler.subT.append(T[i])
                self.actionSampler.subSonT.append(SonT[i])
                self.actionSampler.subRhof.append(hydroVarsInterp[i].energyDensityFalse)
                self.actionSampler.subRhot.append(hydroVarsInterp[i].energyDensityTrue)

                # The integration helper needs three data points before it can be initialised. However, we wait for an
                # additional point so that we can immediately integrate and add it to the true vacuum fraction and false
                # vacuum probability arrays as usual below.
                if integrationHelper_trueVacVol is None and len(self.actionSampler.subT) == 4:
                    integrationHelper_trueVacVol = integration.CubedNestedIntegrationHelper([0, 1, 2],
                        outerFunction_trueVacVol, innerFunction_trueVacVol, sampleTransformationFunction)

                    # Don't add the first element since we have already stored Vext[0] = 0, Pf[0] = 1, etc.
                    for j in range(1, len(integrationHelper_trueVacVol.data)):
                        Vext.append(fourPiOnThree*integrationHelper_trueVacVol.data[j])
                        physicalVolume.append(3 + T[j]*(Vext[-2] - Vext[-1]) / (T[j-1] - T[j]))
                        Pf.append(np.exp(-Vext[-1]))
                        bubbleNumberDensity.append((T[j]/T[j-1])**3 * bubbleNumberDensity[-1]
                            + 0.5*(T[j-1] - T[j])*T[j]**3 * (Gamma[j-1]*Pf[j-1] / (T[j-1]**4 * H[j-1])
                            + Gamma[j]*Pf[j] / (T[j]**4 * H[j])))

                    # Do the same thing for the average bubble radius integration helper. This needs to be done after Pf
                    # has been filled with data because outerFunction_avgBubRad uses this data.
                    integrationHelper_avgBubRad = integration.LinearNestedNormalisedIntegrationHelper([0, 1],
                        outerFunction_avgBubRad, innerFunction_avgBubRad, outerFunction_avgBubRad,
                        sampleTransformationFunction)

                    # Since the average bubble radius integration helper requires one less data point for
                    # initialisation, it currently contains one less data point than it should have. Add one more data
                    # point: the previous temperature sample. The current temperature sample will be added later in this
                    # iteration of i.
                    integrationHelper_avgBubRad.integrate(len(self.actionSampler.subT)-2)

                    for j in range(1, len(integrationHelper_avgBubRad.data)):
                        meanBubbleRadiusArray.append(integrationHelper_avgBubRad.data[j])
                        meanBubbleSeparationArray.append((bubbleNumberDensity[j])**(-1/3))

                #H.append(np.sqrt(self.hubble_squared(T[i])))
                #H.append(np.sqrt(hubble_squared_from_energy_density(rhof[i] - self.groud_state_energy_density)))
                H.append(hydroVarsInterp[i].hubble_constant)
                vw.append(self.getBubbleWallVelocity(hydroVarsInterp[i]))

                Gamma.append(self.calculateGamma(T[i], SonT[i]))

                numBubblesIntegrand.append(Gamma[-1]/(T[i]*H[-1]**4))
                numBubblesCorrectedIntegrand.append(Gamma[-1]*Pf[-1]/(T[i]*H[-1]**4))
                numBubblesIntegral.append(numBubblesIntegral[-1]
                    + 0.5*dT*(numBubblesIntegrand[-1] + numBubblesIntegrand[-2]))
                numBubblesCorrectedIntegral.append(numBubblesCorrectedIntegral[-1]
                    + 0.5*dT*(numBubblesCorrectedIntegrand[-1] + numBubblesCorrectedIntegrand[-2]))

                if integrationHelper_trueVacVol is not None:
                    integrationHelper_trueVacVol.integrate(len(self.actionSampler.subT)-1)
                    Vext.append(fourPiOnThree*integrationHelper_trueVacVol.data[-1])
                    physicalVolume.append(3 + T[i]*(Vext[-2] - Vext[-1]) / (self.actionSampler.subT[-2] - T[i]))
                    Pf.append(np.exp(-Vext[-1]))
                    bubbleNumberDensity.append((T[i]/T[i-1])**3 * bubbleNumberDensity[-1]
                        + 0.5*(T[i-1] - T[i])*T[i]**3 * (Gamma[-2]*Pf[-2] / (T[i-1]**4 * H[-2])
                        + Gamma[-1]*Pf[-1] / (T[i]**4 * H[-1])))
                    # This needs to be done after the new Pf has been added because outerFunction_avgBubRad uses this data.
                    integrationHelper_avgBubRad.integrate(len(self.actionSampler.subT)-1)
                    meanBubbleRadiusArray.append(integrationHelper_avgBubRad.data[-1])
                    meanBubbleSeparationArray.append((bubbleNumberDensity[-1])**(-1/3))

                Tnew = self.actionSampler.subT[-1]
                Tprev = self.actionSampler.subT[-2]
                SonTnew = self.actionSampler.subSonT[-1]
                SonTprev = self.actionSampler.subSonT[-2]

                # TODO: not a great derivative, can do better.
                betaArray.append(H[-1]*Tnew*(SonTprev - SonTnew)/dT)

                # Check if we have reached any milestones (e.g. unit nucleation, percolation, etc.).

                # Unit nucleation (including phantom bubbles).
                if Tn < 0 and numBubblesIntegral[-1] >= 1:
                    interpFactor = (numBubblesIntegral[-1] - 1) / (numBubblesIntegral[-1] - numBubblesIntegral[-2])
                    Tn = Tprev + interpFactor*(Tnew - Tprev)
                    Hn = H[-2] + interpFactor*(H[-1] - H[-2])
                    self.properties.SonTn = SonTprev + (SonTnew - SonTprev) * (numBubblesIntegral[-1] - 1)\
                        / (numBubblesIntegral[-1] - numBubblesIntegral[-2])
                    self.properties.Tn = Tn
                    self.properties.Hn = Hn
                    self.properties.betaTn = Hn*Tn*(SonTprev - SonTnew)/dT
                    # Store the reheating temperature from this point, using conservation of energy.
                    Tn_reh = self.reheat_temperature(Tn)
                    self.properties.Treh_n = Tn_reh

                # Unit nucleation (excluding phantom bubbles).
                if Tnbar < 0 and numBubblesCorrectedIntegral[-1] >= 1:
                    interpFactor = (numBubblesCorrectedIntegral[-1] - 1) / (numBubblesCorrectedIntegral[-1] -
                        numBubblesCorrectedIntegral[-2])
                    Tnbar = Tprev + interpFactor*(Tnew - Tprev)
                    Hnbar = H[-2] + interpFactor*(H[-1] - H[-2])
                    self.properties.SonTnbar = SonTprev + (SonTnew - SonTprev)\
                        * (numBubblesCorrectedIntegral[-1] - 1) / (numBubblesCorrectedIntegral[-1]
                        - numBubblesCorrectedIntegral[-2])
                    self.properties.Tnbar = Tnbar
                    self.properties.Hnbar = Hnbar
                    self.properties.betaTnbar = Hnbar*Tnbar*(SonTprev - SonTnew)/dT
                    # Store the reheating temperature from this point, using conservation of energy.
                    Tnbar_reh = self.reheat_temperature(Tnbar)
                    self.properties.Treh_nbar = Tnbar_reh

                # Percolation.
                if Tp < 0 and Vext[-1] >= percolationThreshold_Vext:
                    indexTp = len(H)-1
                    # max(0, ...) for subcritical transitions, where it is possible that Vext[-2] > percThresh.
                    interpFactor = max(0, (percolationThreshold_Vext - Vext[-2]) / (Vext[-1] - Vext[-2]))
                    Tp = Tprev + interpFactor*(Tnew - Tprev)
                    Hp = H[-2] + interpFactor*(H[-1] - H[-2])
                    self.properties.SonTp = SonTprev + interpFactor*(SonTnew - SonTprev)
                    self.properties.Tp = Tp
                    self.properties.Hp = Hp
                    self.properties.betaTp = Hp*Tp*(SonTprev - SonTnew)/dT

                    # Also store whether the physical volume of the false vacuum was decreasing at Tp.
                    # Make sure to cast to a bool, because JSON doesn't like encoding the numpy.bool type.
                    self.properties.decreasingVphysAtTp = bool(physicalVolume[-1] < 0)

                    # Store the reheating temperature from this point, using conservation of energy.
                    Tp_reh = self.reheat_temperature(Tp)
                    self.properties.Treh_p = Tp_reh

                # Pf = 1/e.
                if Te < 0 and Vext[-1] >= 1:
                    # max(0, ...) for subcritical transitions, where it is possible that Vext[-2] > 1.
                    interpFactor = max(0, (1 - Vext[-2]) / (Vext[-1] - Vext[-2]))
                    Te = Tprev + interpFactor*(Tnew - Tprev)
                    He = H[-2] + interpFactor*(H[-1] - H[-2])
                    self.properties.SonTe = SonTprev + interpFactor*(SonTnew - SonTprev)
                    self.properties.Te = Te
                    self.properties.He = He
                    self.properties.betaTe = He*Te*(SonTprev - SonTnew)/dT

                    # Store the reheating temperature from this point, using conservation of energy.
                    Te_reh = self.reheat_temperature(Te)
                    self.properties.Treh_e = Te_reh

                # Completion.
                if Tf < 0 and Pf[-1] <= completionThreshold:
                    indexTf = len(H)-1
                    if Pf[-1] == Pf[-2]:
                        interpFactor = 0
                    else:
                        interpFactor = (Pf[-1] - completionThreshold)\
                            / (Pf[-1] - Pf[-2])
                    Tf = Tprev + interpFactor*(Tnew - Tprev)
                    Hf = H[-2] + interpFactor*(H[-1] - H[-2])
                    self.properties.SonTf = SonTprev + interpFactor*(SonTnew - SonTprev)
                    self.properties.Tf = Tf
                    self.properties.Hf = Hf
                    self.properties.betaTf = Hf*Tf*(SonTprev - SonTnew)/dT

                    # Also store whether the physical volume of the false vacuum was decreasing at Tf.
                    # Make sure to cast to a bool, because JSON doesn't like encoding the numpy.bool type.
                    self.properties.decreasingVphysAtTf = bool(physicalVolume[-1] < 0)

                    # Store the reheating temperature from this point, using conservation of energy.
                    Tf_reh = self.reheat_temperature(Tf)
                    self.properties.Treh_f = Tf_reh

                    if not self.bAnalyseTransitionPastCompletion:
                        break

                # Physical volume of the false vacuum is decreasing.
                if Ts1 < 0 and physicalVolume[-1] < 0:
                    if physicalVolume[-1] == physicalVolume[-2]:
                        interpFactor = 0
                    else:
                        interpFactor = 1 - physicalVolume[-1] / (physicalVolume[-1] - physicalVolume[-2])
                    Ts1 = Tprev + interpFactor*(Tnew - Tprev)
                    self.properties.TVphysDecr_high = Ts1

                # Physical volume of the false vacuum is increasing *again*.
                if Ts1 > 0 and Ts2 < 0 and physicalVolume[-1] > 0:
                    if physicalVolume[-1] == physicalVolume[-2]:
                        interpFactor = 0
                    else:
                        interpFactor = 1 - physicalVolume[-1] / (physicalVolume[-1] - physicalVolume[-2])
                    Ts2 = Tprev + interpFactor*(Tnew - Tprev)
                    self.properties.TVphysDecr_low = Ts2

            # ==========================================================================================================
            # End subsampling.
            # ==========================================================================================================

            if Tf > 0 and not self.bAnalyseTransitionPastCompletion:
                logger.debug('Found Tf, stopping sampling')
                break

            if sampleData.T <= self.Tmin:
                logger.debug('The transition does not complete before reaching Tmin')
                break

            # Choose the next value of S/T we're aiming to sample.
            success, message = self.actionSampler.getNextSample(sampleData, Gamma, numBubblesIntegral[-1], self.Tmin)

            simIndex += 1

            if self.timeout_phaseHistoryAnalysis > 0:
                endTime = time.perf_counter()
                if endTime - startTime > self.timeout_phaseHistoryAnalysis:
                    return

            if not success:
                logger.debug('Terminating transition analysis after failing to get next action sample. Reason:', message)

                if message in ('Freeze out', 'Reached Tmin'):
                    break

                if precomputedT:
                    self.properties.actionCurveFile = precomputedActionCurveFileName
                self.properties.T = self.actionSampler.T
                self.properties.SonT = self.actionSampler.SonT
                self.properties.error = f'Failed to get next sample at T={sampleData.T}'
                return

        # ==============================================================================================================
        # End transition analysis.
        # ==============================================================================================================

        GammaEff = np.array([Pf[i] * Gamma[i] for i in range(len(Gamma))])

        # Find the maximum nucleation rate to find TGammaMax. Only do so if there is a minimum in the action.
        if TAtSonTmin > 0:
            gammaMaxIndex = np.argmax(Gamma)
            self.properties.TGammaMax = self.actionSampler.subT[gammaMaxIndex]
            self.properties.SonTGammaMax = self.actionSampler.subSonT[gammaMaxIndex]
            self.properties.GammaMax = Gamma[gammaMaxIndex]
            Tlow = self.actionSampler.subT[gammaMaxIndex+1]
            #Tmid = self.actionSampler.subT[gammaMaxIndex]
            Thigh = self.actionSampler.subT[gammaMaxIndex-1]
            Slow = self.actionSampler.subSonT[gammaMaxIndex+1]
            Smid = self.actionSampler.subSonT[gammaMaxIndex]
            Shigh = self.actionSampler.subSonT[gammaMaxIndex-1]
            # TODO: should use proper formula for second derivative with non-uniform grid.
            d2SdT2 = (Shigh - 2*Smid + Slow) / (0.5*(Thigh - Tlow))**2
            logger.debug('Calculating betaV, d2SdT2 = {}', d2SdT2)
            if d2SdT2 > 0:
                self.properties.betaV = H[gammaMaxIndex]*self.actionSampler.subT[gammaMaxIndex]*np.sqrt(d2SdT2)

        meanBubbleSeparation = (bubbleNumberDensity[indexTp])**(-1/3)

        if self.bReportAnalysis:
            Tn = self.properties.Tn
            Tp = self.properties.Tp
            Te = self.properties.Te
            Tf = self.properties.Tf
            Ts1 = self.properties.TVphysDecr_high
            Ts2 = self.properties.TVphysDecr_low
            Tp_reh = self.properties.Treh_p
            Te_reh = self.properties.Treh_e
            Tf_reh = self.properties.Treh_f

            print('-------------------------------------------------------------------------------------------')

            print('N(T_0): ', numBubblesIntegral[-1])
            print('T_0: ', self.actionSampler.subT[-1])
            print('Tc: ', self.properties.Tc)
            if Tn > 0:
                print(f'Unit nucleation at T = {Tn}, where S = {self.properties.SonTn}')
            else:
                print('No unit nucleation.')
            if Tp > 0:
                print(f'Percolation at T = {Tp}, where S = {self.properties.SonTp}')
            else:
                print('No percolation. Terminated analysis at T =', self.actionSampler.subT[-1], 'where Pf =', Pf[-1])
            if Te > 0:
                print(f'Vext = 1 at T = {Te}, where S = {self.properties.SonTe}')
            if Tf > 0:
                print(f'Completion at T = {Tf}, where S = {self.properties.SonTf}')
            else:
                if Tp > 0:
                    print('No completion. Terminated analysis at T =', self.actionSampler.subT[-1], 'where Pf =',
                        Pf[-1])
                else:
                    print('No completion.')
            if Ts1 > 0:
                if Ts2 > 0:
                    print('Physical volume of the false vacuum decreases between', Ts2, 'and', Ts1)
                else:
                    print('Physical volume of the false vacuum decreases below', Ts1)
            if Tn > 0:
                print(f'Reheating from unit nucleation: Treh(Tn): {Tn} -> {Tn_reh}')
            if Tp > 0:
                print(f'Reheating from percolation: Treh(Tp): {Tp} -> {Tp_reh}')
            if Te > 0:
                print(f'Reheating from Vext=1: Treh(Te): {Te} -> {Te_reh}')
            if Tf > 0:
                print(f'Reheating from completion: Treh(Tf): {Tf} -> {Tf_reh}')
            if TAtSonTmin > 0:
                print('Action minimised at T =', TAtSonTmin)
            if self.properties.TGammaMax > 0:
                print('Nucleation rate maximised at T =', self.properties.TGammaMax)

            print('Mean bubble separation:', meanBubbleSeparation)

            print('-------------------------------------------------------------------------------------------')

        if self.bDebug:
            print('Total action evaluations:', totalActionEvaluations)

        if self.bPlot:
            plt.rcParams.update({"text.usetex": True})

            plt.plot(self.actionSampler.subT, vw)
            plt.xlabel('T')
            plt.ylabel('$v_w$')
            plt.ylim(0, 1)
            plt.show()

            if Tf > 0:
                maxIndex = len(self.actionSampler.subT)
                maxIndex = min(len(self.actionSampler.subT)-1, maxIndex - (maxIndex - indexTf)//2)
                physicalVolumeRelative = [100 * (Tf/self.actionSampler.subT[i])**3 * Pf[i]
                    for i in range(maxIndex+1)]

                ylim = np.array(physicalVolumeRelative[:min(indexTf+1, maxIndex)]).max(initial=1.)*1.2

                textXOffset = 0.01*(self.actionSampler.subT[0] - self.actionSampler.subT[maxIndex])
                textY = 0.1

                plt.figure(figsize=(14, 11))
                plt.plot(self.actionSampler.subT[:maxIndex+1], physicalVolumeRelative, zorder=3, lw=3.5)

                if Ts1 > 0 and Ts2 > 0: plt.axvspan(Ts2, Ts1, alpha=0.3, color='r', zorder=-1)
                if Tp > 0:
                    plt.axvline(Tp, c='g', ls='--', lw=2)
                    plt.text(Tp + textXOffset, textY, '$T_p$', fontsize=44, horizontalalignment='left')
                if Te > 0:
                    plt.axvline(Te, c='b', ls='--', lw=2)
                    plt.text(Te + textXOffset, textY, '$T_e$', fontsize=44, horizontalalignment='left')
                if Tf > 0:
                    plt.axvline(Tf, c='k', ls='--', lw=2)
                    plt.text(Tf - textXOffset, textY, '$T_f$', fontsize=44, horizontalalignment='right')
                plt.axhline(1., c='gray', ls=':', lw=2, zorder=-1)
                plt.xlabel('$T \,\, \\mathrm{[GeV]}$', fontsize=52)
                plt.ylabel('$\\mathcal{V}_{\\mathrm{phys}}(T)/\\mathcal{V}_{\\mathrm{phys}}(T_f)$', fontsize=52,
                    labelpad=20)
                plt.xlim(0, self.actionSampler.subT[0])
                plt.ylim(0, ylim)
                plt.tick_params(size=10, labelsize=40)
                plt.margins(0, 0)
                plt.tight_layout()
                plt.show()
                #saveFolder = 'C:/Work/Monash/PhD/Documents/Subtleties of supercooled cosmological first-order phase transitions/images/'
                #plt.savefig(saveFolder + 'Relative physical volume.png', bbox_inches='tight', pad_inches=0.1)

            ylim = np.array(physicalVolume).min(initial=0.)
            ylim *= 1.2 if ylim < 0 else 0.8
            if -0.5 < ylim < 0:
                ylim = -0.5

            textXOffset = 0.01*(self.actionSampler.subT[0] - 0)
            textY = ylim + 0.07*(3.5 - ylim)

            plt.figure(figsize=(14, 11))
            plt.plot(self.actionSampler.subT, physicalVolume, zorder=3, lw=3.5)
            #if TVphysDecr_high > 0: plt.axvline(TVphysDecr_high, c='r', ls='--', lw=2)
            #if TVphysDecr_low > 0: plt.axvline(TVphysDecr_low, c='r', ls='--', lw=2)
            if Ts1 > 0 and Ts2 > 0: plt.axvspan(Ts2, Ts1, alpha=0.3, color='r', zorder=-1)
            if Tp > 0:
                plt.axvline(Tp, c='g', ls='--', lw=2)
                plt.text(Tp + textXOffset, textY, '$T_p$', fontsize=44, horizontalalignment='left')
            if Te > 0:
                plt.axvline(Te, c='b', ls='--', lw=2)
                plt.text(Te + textXOffset, textY, '$T_e$', fontsize=44, horizontalalignment='left')
            if Tf > 0:
                plt.axvline(Tf, c='k', ls='--', lw=2)
                plt.text(Tf - textXOffset, textY, '$T_f$', fontsize=44, horizontalalignment='right')
            plt.axhline(3., c='gray', ls=':', lw=2, zorder=-1)
            plt.axhline(0., c='gray', ls=':', lw=2, zorder=-1)
            plt.xlabel('$T \,\, \\mathrm{[GeV]}$', fontsize=50)
            plt.ylabel('$\\frac{\\displaystyle d}{\\displaystyle dt} \\mathcal{V}_{\\mathrm{phys}} \,\, \\mathrm{[GeV]}$',
                fontsize=50, labelpad=20)
            plt.xlim(0, self.actionSampler.subT[0])
            plt.ylim(ylim, 3.5)
            plt.tick_params(size=10, labelsize=40)
            plt.margins(0, 0)
            plt.tight_layout()
            plt.show()
            #saveFolder = 'C:/Work/Monash/PhD/Documents/Subtleties of supercooled cosmological first-order phase transitions/images/'
            #plt.savefig(saveFolder + 'Decreasing physical volume.png', bbox_inches='tight', pad_inches=0.1)

        if self.properties.Tp > 0:
            #transitionStrength, _, _ = calculateTransitionStrength(self.potential, self.fromPhase, self.toPhase,
            #    self.properties.Tp)
            # TODO: do this elsewhere, maybe a thermal params file.
            hydroVars = hydrodynamics.make_hydro_vars(self.fromPhase, self.toPhase, self.potential, Tp)  # TODO why not pass vacuum energy
            thetaf = (hydroVars.energyDensityFalse - hydroVars.pressureFalse/hydroVars.soundSpeedSqTrue) / 4
            thetat = (hydroVars.energyDensityTrue - hydroVars.pressureTrue/hydroVars.soundSpeedSqTrue) / 4
            transitionStrength = 4*(thetaf - thetat) / (3*hydroVars.enthalpyDensityFalse)

            #energyWeightedBubbleRadius, volumeWeightedBubbleRadius = calculateTypicalLengthScale(transition.Tp, indexTp,
            #    indexTn, transition.Tc, meanBubbleSeparation, vw, H, actionSampler.subRhoV, GammaEff, actionSampler.subT,
            #    bPlot=bPlot, bDebug=bDebug)

            self.properties.transitionStrength = transitionStrength
            self.properties.meanBubbleRadius = meanBubbleRadiusArray[indexTp]
            self.properties.meanBubbleSeparation = meanBubbleSeparation
            #transition.energyWeightedBubbleRadius = energyWeightedBubbleRadius
            #transition.volumeWeightedBubbleRadius = volumeWeightedBubbleRadius

            if self.bDebug:
                print('Transition strength:', transitionStrength)

        # If the transition has completed before Teq was identified, predict the value of Teq through extrapolation.
        # TODO: don't worry about Teq at the moment (29/08/2023).
        """if Teq == 0 and len(self.actionSampler.T) > 2:
            Teq = self.predictTeq()
            self.properties.Teq = Teq
            # Don't worry about SonTeq."""

        if self.bReportAnalysis:
            print('Mean bubble separation (Tp):', meanBubbleSeparationArray[indexTp])
            print('Average bubble radius (Tp): ', meanBubbleRadiusArray[indexTp])
            print('Hubble radius (Tp):         ', 1/H[indexTp])
            print('Mean bubble separation (Tf):', meanBubbleSeparationArray[indexTf])
            print('Average bubble radius (Tf): ', meanBubbleRadiusArray[indexTf])
            print('Hubble radius (Tf):         ', 1/H[indexTf])

        if self.bPlot:
            Tn = self.properties.Tn
            Tp = self.properties.Tp
            Te = self.properties.Te
            Tf = self.properties.Tf
            SonTn = self.properties.SonTn
            SonTp = self.properties.SonTp
            SonTe = self.properties.SonTe
            SonTf = self.properties.SonTf
            #plt.rcParams.update({"text.usetex": True})
            #plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']  # for \text command

            plt.figure(figsize=(12, 8))
            plt.plot(self.actionSampler.subT, Gamma)
            plt.plot(self.actionSampler.subT, GammaEff)
            #plt.plot(actionSampler.subT, approxGamma)
            #plt.plot(actionSampler.subT, taylorExpGamma)
            plt.xlabel('$T$', fontsize=24)
            plt.ylabel('$\\Gamma(T)$', fontsize=24)
            if self.properties.TGammaMax > 0: plt.axvline(self.properties.TGammaMax, c='g', ls=':')
            if self.properties.Tmin > 0: plt.axvline(self.properties.Tmin, c='r', ls=':')
            if Tp > 0: plt.axvline(Tp, c='g', ls=':')
            if Te > 0: plt.axvline(Te, c='b', ls=':')
            if Tf > 0: plt.axvline(Tf, c='k', ls=':')
            plt.legend(['$\\mathrm{standard}$', '$\\mathrm{effective}$'], fontsize=24)
            plt.tick_params(size=5, labelsize=16)
            plt.margins(0, 0)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(12, 8))
            plt.plot(self.actionSampler.subT, bubbleNumberDensity, linewidth=2.5)
            plt.xlabel('$T \, \\mathrm{[GeV]}$', fontsize=24)
            plt.ylabel('$n_B(T)$', fontsize=24)
            if Tn > 0: plt.axvline(Tn, c='r', ls=':')
            if Tp > 0: plt.axvline(Tp, c='g', ls=':')
            if Te > 0: plt.axvline(Te, c='b', ls=':')
            if Tf > 0: plt.axvline(Tf, c='k', ls=':')
            plt.tick_params(size=5, labelsize=16)
            plt.margins(0, 0)
            plt.tight_layout()
            plt.show()

            highTempIndex = 1

            # Search for the highest temperature for which the mean bubble separation is not larger than it is at the
            # lowest sampled temperature.
            for i in range(1, len(self.actionSampler.subT)):
                if meanBubbleSeparationArray[i] <= meanBubbleSeparationArray[-1]:
                    highTempIndex = i
                    break

            # If the mean bubble separation is always larger than it is at the lowest sampled temperature, plot the
            # entire range of sampled temperatures.
            if highTempIndex == len(self.actionSampler.subT)-1:
                highTempIndex = 0

            plt.figure(figsize=(12, 8))
            plt.plot(self.actionSampler.subT, meanBubbleRadiusArray, linewidth=2.5)
            plt.plot(self.actionSampler.subT, meanBubbleSeparationArray, linewidth=2.5)
            plt.xlabel('$T \, \\mathrm{[GeV]}$', fontsize=24)
            #plt.ylabel('$\\overline{R}_B(T)$', fontsize=24)
            plt.legend(['$\\overline{R}_B(T)$', '$R_*(T)$'], fontsize=24)
            if Tn > 0: plt.axvline(Tn, c='r', ls=':')
            if Tp > 0: plt.axvline(Tp, c='g', ls=':')
            if Te > 0: plt.axvline(Te, c='b', ls=':')
            if Tf > 0: plt.axvline(Tf, c='k', ls=':')
            plt.xlim(self.actionSampler.subT[-1], self.actionSampler.subT[highTempIndex])
            plt.ylim(0, 1.2*max(meanBubbleSeparationArray[-1], meanBubbleRadiusArray[-1]))
            plt.tick_params(size=5, labelsize=16)
            plt.margins(0, 0)
            plt.tight_layout()
            plt.show()

            highTempIndex = 0
            lowTempIndex = len(self.actionSampler.SonT)

            minAction = min(self.actionSampler.SonT)
            maxAction = self.actionSampler.maxSonTThreshold

            # Search for the lowest temperature for which the action is not significantly larger than the maximum
            # significant action.
            for i in range(len(self.actionSampler.SonT)):
                if self.actionSampler.SonT[i] <= maxAction:
                    highTempIndex = i
                    break

            # Search for the lowest temperature for which the action is not significantly larger than the maximum
            # significant action.
            for i in range(len(self.actionSampler.SonT)-1, -1, -1):
                if self.actionSampler.SonT[i] <= maxAction:
                    lowTempIndex = i
                    break

            plt.figure(figsize=(12,8))
            plt.plot(self.actionSampler.subT, self.actionSampler.subSonT, linewidth=2.5)
            #plt.plot(actionSampler.subT, approxSonT, linewidth=2.5)
            plt.scatter(self.actionSampler.T, self.actionSampler.SonT)
            if Tn > -1:
                plt.axvline(Tn, c='r', ls=':')
                plt.axhline(SonTn, c='r', ls=':')
            if Tp > -1:
                plt.axvline(Tp, c='g', ls=':')
                plt.axhline(SonTp, c='g', ls=':')
            if Te > -1:
                plt.axvline(Te, c='b', ls=':')
                plt.axhline(SonTe, c='b', ls=':')
            if Tf > -1:
                plt.axvline(Tf, c='k', ls=':')
                plt.axhline(SonTf, c='k', ls=':')
            plt.minorticks_on()
            plt.grid(visible=True, which='major', color='k', linestyle='--')
            plt.grid(visible=True, which='minor', color='gray', linestyle=':')
            plt.xlabel('$T \, \\mathrm{[GeV]}$', fontsize=24)
            plt.ylabel('$S(T)$', fontsize=24)
            #plt.legend(['precise', 'approx'])
            plt.xlim(self.actionSampler.T[lowTempIndex], self.actionSampler.T[highTempIndex])
            plt.ylim(minAction - 0.05*(maxAction - minAction), maxAction)
            plt.tick_params(size=5, labelsize=16)
            plt.margins(0, 0)
            plt.tight_layout()
            plt.show()

            # Number of bubbles plotted over entire sampled temperature range, using log scale for number of bubbles.
            plt.figure(figsize=(12, 8))
            plt.plot(self.actionSampler.subT, numBubblesCorrectedIntegral, linewidth=2.5)
            plt.plot(self.actionSampler.subT, numBubblesIntegral, linewidth=2.5)
            #plt.plot(actionSampler.subT, numBubblesApprox, linewidth=2.5)
            if Tn > 0: plt.axvline(Tn, c='r', ls=':')
            if Tp > 0: plt.axvline(Tp, c='g', ls=':')
            if Te > 0: plt.axvline(Te, c='b', ls=':')
            if Tf > 0: plt.axvline(Tf, c='k', ls=':')
            plt.xlabel('$T \, \\mathrm{[GeV]}$', fontsize=24)
            #plt.ylabel('$N(T)$', fontsize=24)
            plt.yscale('log')
            #plt.legend(['precise', 'approx'])
            plt.legend(['$N(T)$', '$N^{\\mathrm{ext}}(T)$'], fontsize=24)
            plt.tick_params(size=5, labelsize=16)
            plt.margins(0, 0)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(12, 8))
            plt.plot(self.actionSampler.subT, Pf, linewidth=2.5)
            if Tn > 0: plt.axvline(Tn, c='r', ls=':')
            if Tp > 0: plt.axvline(Tp, c='g', ls=':')
            if Te > 0: plt.axvline(Te, c='b', ls=':')
            if Tf > 0: plt.axvline(Tf, c='k', ls=':')
            plt.xlabel('$T \, \\mathrm{[GeV]}$', fontsize=40)
            plt.ylabel('$P_f(T)$', fontsize=40)
            plt.tick_params(size=8, labelsize=28)
            plt.margins(0,0)
            plt.tight_layout()
            plt.show()
            #plt.savefig("output/plots/Pf_vs_T_BP2.pdf")
            #plt.savefig('E:/Monash/PhD/Milestones/Confirmation Review/images/xSM P(T) vs T.png', bbox_inches='tight',
            #    pad_inches=0.05)

        if len(precomputedT) > 0:
            self.properties.actionCurveFile = precomputedActionCurveFileName
        self.properties.T = self.actionSampler.T
        self.properties.SonT = self.actionSampler.SonT
        self.properties.totalNumBubbles = numBubblesIntegral[-1]
        self.properties.totalNumBubblesCorrected = numBubblesCorrectedIntegral[-1]

        if self.bComputeSubsampledThermalParams:
            self.properties.TSubampleArray = self.actionSampler.subT
            self.properties.HArray = H
            self.properties.betaArray = betaArray
            self.properties.meanBubbleSeparationArray = meanBubbleSeparationArray
            self.properties.meanBubbleRadiusArray = meanBubbleRadiusArray
            self.properties.Pf = Pf

    def primeTransitionAnalysis(self, startTime: float) -> (Optional[ActionSample], list[ActionSample]):
        TcData = ActionSample(self.properties.Tc)

        if self.bDebug:
            print(f'Bisecting to find S/T = {self.actionSampler.maxSonTThreshold} ± {self.actionSampler.toleranceSonT}')

        # Use bisection to find the temperature at which S/T ~ maxSonTThreshold.
        actionCurveShapeAnalysisData = self.findNucleationTemperatureWindow_refined(startTime=startTime)

        if self.timeout_phaseHistoryAnalysis > 0:
            endTime = time.perf_counter()
            if endTime - startTime > self.timeout_phaseHistoryAnalysis:
                return None, []

        # TODO: Maybe use the names from here anyway? The current set of names is a little confusing!
        data = actionCurveShapeAnalysisData.desiredData
        bisectMinData = actionCurveShapeAnalysisData.nextBelowDesiredData
        lowerSonTData = actionCurveShapeAnalysisData.storedLowerActionData
        allSamples = actionCurveShapeAnalysisData.actionSamples
        bBarrierAtTmin = actionCurveShapeAnalysisData.bBarrierAtTmin

        # If we didn't find any action values near the nucleation threshold, we are done.
        if data is None or not data.is_valid:
            if len(allSamples) == 0:
                if self.bReportAnalysis:
                    print('No transition')
                    print('No action samples')
                return None, []

            allSamples = np.array(allSamples)
            minSonTIndex = np.argmin(allSamples[:, 1])


            if self.bReportAnalysis:
                print('No transition')
                print('Lowest sampled S/T =', allSamples[minSonTIndex, 1], 'at T =', allSamples[minSonTIndex, 0])

            # if bDebug or bPlot:
            # This is a little confusing at first, but what this does is:
            # - allSamples.argsort(axis=0): sort the indices of both columns (T and S/T) based on the values stored in
            #   the corresponding positions.
            # - [:,0]: grab the first column (the T column).
            # - Use the sorted indices based on T to grab the elements of allSamples in the correct order.
            # This returns a copy of allSamples with the same (T, S/T) tuples, but sorted by the T values.
            allSamples = allSamples[allSamples.argsort(axis=0)[:, 0]]
            # TODO: couldn't we just use sort(key=)?

            T = [sample[0] for sample in allSamples]
            SonT = [sample[1] for sample in allSamples]

            if self.bDebug:
                print('T:', T)
                print('SonT:', SonT)

            if self.bPlot:
                plt.plot(T, SonT)
                plt.xlabel('$T$')
                plt.ylabel('$S_3/T$')
                plt.yscale('log')
                # plt.ylim(bottom=2)
                plt.show()

            self.properties.T = T
            self.properties.SonT = SonT
            return None, []
        else:
            if self.bPlot:
                print(f'Data: (T: {data.T}, S/T: {data.SonT})')
                print('len(lowerSonTData):', len(lowerSonTData))

        intermediateData = data.copy()

        self.actionSampler.lowerSonTData = lowerSonTData

        if self.bDebug:
            print('Attempting to find next reasonable (T, S/T) sample below maxSonTThreshold...')

        if data.SonT > self.actionSampler.minSonTThreshold + 0.8 * (
                self.actionSampler.maxSonTThreshold - self.actionSampler.minSonTThreshold):
            if self.bDebug:
                print('Presumably not a subcritical transition curve, with current S/T near maxSonTThreshold.')
            subcritical = False
            # targetSonT is the first S/T value we would like to sample. Skipping this might lead to numerical errors in the
            # integration, and sampling at higher S/T values is numerically insignificant.
            # targetSonT = minSonTThreshold + 0.98*(min(maxSonTThreshold, data.SonT) - minSonTThreshold)
            targetSonT = data.SonT * self.actionSampler.stepSizeMax

            # Check if the bisection window can inform which temperature we should sample to reach the target S/T.
            if abs(bisectMinData.SonT - targetSonT) < 0.3 * (
                    self.actionSampler.maxSonTThreshold - self.actionSampler.minSonTThreshold):
                interpFactor = (targetSonT - bisectMinData.SonT) / (data.SonT - bisectMinData.SonT)
                intermediateData.T = bisectMinData.T + interpFactor * (data.T - bisectMinData.T)
            # Otherwise, check if the low S/T data can inform which temperature we should sample to reach the target S/T.
            elif len(lowerSonTData) > 0 and abs(lowerSonTData[-1].SonT - targetSonT) \
                    < 0.5 * (self.actionSampler.maxSonTThreshold - self.actionSampler.minSonTThreshold):
                interpFactor = (targetSonT - lowerSonTData[-1].SonT) / (data.SonT - lowerSonTData[-1].SonT)
                intermediateData.T = lowerSonTData[-1].T + interpFactor * (data.T - lowerSonTData[-1].T)
            # Otherwise, all that's left to do is guess.
            else:
                # Try sampling S/T at a temperature just below where S/T = maxSonTThreshold, and determine where to sample
                # next based on the result.
                intermediateData.T = self.Tmin + 0.99*(data.T - self.Tmin)

            self.actionSampler.evaluateAction(intermediateData)

            if self.timeout_phaseHistoryAnalysis > 0:
                endTime = time.perf_counter()
                if endTime - startTime > self.timeout_phaseHistoryAnalysis:
                    return None, []

            if not intermediateData.is_valid:
                self.properties.error = f'Failed to evaluate action at trial temperature T={intermediateData.T}'
                return None, []

            # If we happen to sample too far away from 'data' (which can happen if S/T is very steep near maxSonTThreshold),
            # then we should correct our next sample to be closer to maxSonTThreshold. In case of a noisy action, make sure
            # to limit the number of samples and simply choose the result with the closest S/T to maxSonTThreshold.
            maxCorrectionSamples = 5
            correctionSamplesTaken = 0
            closestPoint = intermediateData.copy()

            # While our sample's S/T is too far from the target value, step closer to 'data' and try again.
            while correctionSamplesTaken < maxCorrectionSamples \
                    and abs(1 - abs(intermediateData.SonT - data.SonT) / data.SonT) < self.actionSampler.stepSizeMax:
                if self.bDebug:
                    print('Sample too far from target S/T value at T =', intermediateData.T, 'with S/T =',
                          intermediateData.SonT)
                    print('Trying again at T =', 0.5*(intermediateData.T + data.T))
                correctionSamplesTaken += 1
                # Step halfway across the interval and try again.
                intermediateData.T = 0.5*(intermediateData.T + data.T)
                self.actionSampler.evaluateAction(intermediateData)

                if self.timeout_phaseHistoryAnalysis > 0:
                    endTime = time.perf_counter()
                    if endTime - startTime > self.timeout_phaseHistoryAnalysis:
                        return None, []

                # Store this point so we don't have to resample near it.
                self.actionSampler.storeLowerSonTData(intermediateData.copy())

                # If this is the closest point, store this in case the next sample is worse (for a noisy action).
                if abs(intermediateData.SonT - data.SonT) < abs(closestPoint.SonT - data.SonT):
                    closestPoint = intermediateData.copy()

            # If we corrected intermediate data, make sure to update it to the closest point (to maxSonTThreshold) sampled.
            if correctionSamplesTaken > 0:
                closestPoint = intermediateData.copy()

            # Given that we took a small step in temperature and have a relatively large S/T, an increase in S/T means there
            # is insufficient time for nucleation to occur. It is improbable that S/T would drop to a small enough value
            # within this temperature range to yield nucleation.
            if intermediateData.SonT >= data.SonT:
                if self.bDebug:
                    print('S/T increases before nucleation can occur.')
                return None, self.actionSampler.lowerSonTData

            self.actionSampler.T.extend([data.T, intermediateData.T])
            self.actionSampler.SonT.extend([data.SonT, intermediateData.SonT])
        else:
            if self.bDebug:
                print(
                    'Presumably a subcritical transition curve, with current S/T significantly below maxSonTThreshold.')
            subcritical = True

            # Take a very small step, with the size decreasing as S/T decreases.
            interpFactor = 0.99 + 0.009*(1 - data.SonT/self.actionSampler.minSonTThreshold)
            intermediateData.T = self.Tmin + interpFactor*(data.T - self.Tmin)

            # There's no point taking a tiny temperature step if we already took a larger step away from Tmax as our highest
            # sample point.
            intermediateData.T = min(intermediateData.T, 2*data.T - self.Tmax)
            self.actionSampler.evaluateAction(intermediateData)

            if self.timeout_phaseHistoryAnalysis > 0:
                endTime = time.perf_counter()
                if endTime - startTime > self.timeout_phaseHistoryAnalysis:
                    return None, []

            # Don't accept cases where S/T is negative (translated to 0) for the last two evaluations. This suggests the
            # bounce solver is failing and we cannot proceed reliably.
            if not intermediateData.is_valid or intermediateData.SonT == 0 and TcData.SonT == 0:
                self.properties.error = f'Failed to evaluate action at trial temperature T={intermediateData.T}'
                # print('This was for a subcritical transition with initial S/T:', data.SonT, 'at T:', data.T)
                return None, []

            # If we couldn't sample all the way to Tmax, predict what the action would be at Tmax and store that as a
            # previous sample to be used in the following integration. intermediateData will be used as the next sample
            # point.
            if data.T < self.Tmax:
                maxSonT = data.SonT + (data.SonT - intermediateData.SonT) * (self.Tmax - data.T) / (
                            data.T - intermediateData.T)

                self.actionSampler.T.extend([self.Tmax, data.T])
                self.actionSampler.SonT.extend([maxSonT, data.SonT])

                # We have already sampled this data point and should use it as the next point in the integration. Storing it
                # in lowerSonTData automatically results in this desired behaviour. No copy is required as we don't alter
                # intermediateData from this point on.
                self.actionSampler.storeLowerSonTData(intermediateData)
            # If we sampled all the way to Tmax, use the sample there and intermediateData as the samples for the following
            # integration.
            else:
                self.actionSampler.T.extend([data.T, intermediateData.T])
                self.actionSampler.SonT.extend([data.SonT, intermediateData.SonT])

        if self.bDebug:
            print('Found next sample: T =', intermediateData.T, 'and S/T =', intermediateData.SonT)

        # Now take the same step in temperature and evaluate the action again.
        sampleData = intermediateData.copy()
        sampleData.T = 2 * intermediateData.T - data.T

        if not subcritical and len(self.actionSampler.lowerSonTData) > 0 and self.actionSampler.lowerSonTData[
            -1].T >= sampleData.T:
            if self.actionSampler.lowerSonTData[-1].T < self.actionSampler.T[-1]:
                # sampleData.T = actionSampler.lowerSonTData[-1].T
                # sampleData.action = actionSampler.lowerSonTData[-1].action
                # sampleData.SonT = actionSampler.lowerSonTData[-1].SonT
                sampleData = self.actionSampler.lowerSonTData[-1].copy()
            else:
                self.actionSampler.evaluateAction(sampleData)

                if self.timeout_phaseHistoryAnalysis > 0:
                    endTime = time.perf_counter()
                    if endTime - startTime > self.timeout_phaseHistoryAnalysis:
                        return None, []

            self.actionSampler.lowerSonTData.pop()
        else:
            sampleData.action = -1
            sampleData.SonT = -1

            self.actionSampler.evaluateAction(sampleData)

            if self.timeout_phaseHistoryAnalysis > 0:
                endTime = time.perf_counter()
                if endTime - startTime > self.timeout_phaseHistoryAnalysis:
                    return None, []

            if not sampleData.is_valid:
                print('Failed to evaluate action at trial temperature T=', sampleData.T)

        self.actionSampler.calculateStepSize(low=sampleData, mid=intermediateData, high=data)

        # We have already sampled this data point and should use it as the next point in the integration. Storing it in
        # lowerSonTData automatically results in this desired behaviour. For a near-instantaneous subcritical transition,
        # this will actually be the *second* next point in the integration, as the handling of intermediateData is also
        # postponed, and should be done before sampleData.
        self.actionSampler.storeLowerSonTData(sampleData.copy())

        return sampleData, allSamples

    def estimateMaximumSignificantSonT(self, tolerance=2.0):
        actionMin = 50
        actionMax = 200
        action = 160

        while actionMax - actionMin > tolerance:
            action = 0.5*(actionMin + actionMax)

            nucRate = self.calculateInstantaneousNucleationRate(self.Tmax, action)
            numBubbles = nucRate*(self.Tmax - self.Tmin)

            if 0.1 < numBubbles < 10:
                return action
            elif numBubbles < 1:
                actionMax = action
            else:
                actionMin = action

        return action

    def findNucleationTemperatureWindow_refined(self, startTime: float = -1.0):
        actionCurveShapeAnalysisData = ActionCurveShapeAnalysisData()
        Tstep = 0.01*(self.Tmax - self.Tmin)
        actionSamples = []
        lowerActionData = []

        TLow = self.Tmin
        THigh = self.Tmax

        # TODO: come up with a new name for this now that evaluateAction has replaced calculateActionSimple.
        def evaluateAction_internal():
  
            self.actionSampler.evaluateAction(data)
      

            actionSamples.append((data.T, data.SonT, data.is_valid))

            if data.is_valid and self.actionSampler.minSonTThreshold < data.SonT < self.actionSampler.maxSonTThreshold:
                lowerActionData.append(data.copy())

            return True

        def getSignLabel():
            if not data.is_valid:
                return 'unknown'
            elif abs(data.SonT - self.actionSampler.maxSonTThreshold) <= self.actionSampler.toleranceSonT:
                return 'equal'
            elif data.SonT > self.actionSampler.maxSonTThreshold:
                return 'above'
            else:
                return 'below'

        def saveCurveShapeAnalysisData(success=False):
            dataToUse = data
            lowerTSampleToUse = lowerTSample

            # If the transition is said to fail (e.g. due to failure to evaluate the bounce in the thin-walled limit), check
            # if we have enough data to suggest the transition should occur.
            if not success:
                # Find the highest temperature for which the action is below the nucleation threshold.
                maxTempBelowIndex = -1
                prevMaxTempBelowIndex = -1
                for i in range(len(actionSamples)):
                    if actionSamples[i][2] and actionSamples[i][1] < self.actionSampler.maxSonTThreshold:
                        if maxTempBelowIndex < 0 or actionSamples[i][0] > actionSamples[maxTempBelowIndex][0]:
                            prevMaxTempBelowIndex = maxTempBelowIndex
                            maxTempBelowIndex = i

                # If such a sample exists, we can use this as the desired data point.
                if maxTempBelowIndex > -1:
                    success = True
                    temp = actionSamples[maxTempBelowIndex][0]
                    action = actionSamples[maxTempBelowIndex][1]
                    dataToUse = ActionSample(temp, action*temp)
                    dataToUse.is_valid = True

                    # Attempt to also store the sample point directly before this in temperature.
                    if prevMaxTempBelowIndex > -1:
                        temp = actionSamples[prevMaxTempBelowIndex][0]
                        action = actionSamples[prevMaxTempBelowIndex][1]
                        lowerTSampleToUse = ActionSample(temp, action*temp)
                        lowerTSampleToUse.is_valid = True
                    else:
                        lowerTSampleToUse = None

            if success:
                actionCurveShapeAnalysisData.copyDesiredData(dataToUse)
                actionCurveShapeAnalysisData.copyNextBelowDesiredData(lowerTSampleToUse)
                # We probably just stored data in lowerActionData, so remove it since we're using it as the target value.
                if self.actionSampler.minSonTThreshold < dataToUse.SonT < self.actionSampler.maxSonTThreshold:
                    lowerActionData.pop()
            actionCurveShapeAnalysisData.copyActionSamples(actionSamples)
            actionCurveShapeAnalysisData.copyStoredLowerActionData(lowerActionData)

        # Evaluate the action at Tmin. (Actually just above so we avoid issues where the phase might disappear.)
        data = ActionSample(self.Tmin+Tstep)
        lowerTSample = ActionSample(self.Tmin+Tstep)
        bSuccess = evaluateAction_internal()

        if not bSuccess:
            saveCurveShapeAnalysisData(False)
            return actionCurveShapeAnalysisData

        labelLow = getSignLabel()

        # TODO: the assumption is that if the action calculation fails, it's probably because the barrier becomes too small,
        #  hence the action would be very low.
        if labelLow == 'unknown':
            if self.bAllowErrorsForTn:
                labelLow = 'below'
            # else we would have returned already.

        # If the action is high at Tmin.
        if labelLow == 'above' or labelLow == 'equal':
            # Check at a slightly higher temperature to determine whether the action is increasing or decreasing.

            # Store the result at Tmin to compare to.
            minT = data.T
            minAction = data.SonT
            minLabel = labelLow

            lowerTSample = data.copy()
            data.T += Tstep
            bSuccess = evaluateAction_internal()

            if not bSuccess:
                saveCurveShapeAnalysisData(False)
                return actionCurveShapeAnalysisData

            if data.is_valid:
                labelLow = getSignLabel()
                TLow = data.T

                # If the action is increasing (going from Tmin to Tmin+deltaT), then we will not find a lower action.
                if data.SonT > minAction:
                    # If the action at the minimum temperature was equal to the threshold within tolerance, we have shown
                    # that it was the minimum action overall and can return it as the only solution.
                    if minLabel == 'equal':
                        data.T = minT
                        data.SonT = minAction
                        actionCurveShapeAnalysisData.copyDesiredData(data)
                        

                    saveCurveShapeAnalysisData()
                    return actionCurveShapeAnalysisData

        # Evaluate the action at Tmax if this transition is not evaluated at the critical temperature. Otherwise we already
        # know the action is divergent at the critical temperature.
        if self.properties.Tc and (self.Tmax < self.properties.Tc or self.properties.subcritical):
            if self.Tmax < self.properties.Tc:
                data.T = self.Tmax
            else:
                data.T = self.Tmin + 0.98*(self.Tmax - self.Tmin)

            bSuccess = evaluateAction_internal()

            if not bSuccess:
                saveCurveShapeAnalysisData(False)
                return actionCurveShapeAnalysisData

            labelHigh = getSignLabel()

            seenBelowAtTHigh = labelHigh == 'below'

            # If the action at the high temperature is too low, sample closer to Tmax.
            while labelHigh == 'below' and abs(self.Tmax - data.T)/self.Tmax > 1e-4:
                lowerTSample = data.copy()
                data.T = 0.5*(data.T + self.Tmax)
                Tstep = 0.1*(self.Tmax - data.T)
                bSuccess = evaluateAction_internal()

                if not bSuccess:
                    saveCurveShapeAnalysisData(False)
                    return actionCurveShapeAnalysisData

                # TODO: cleanup if this still works [2023]
                #calculateActionSimple(potential, data, fromPhase, toPhase, Tstep=-Tstep, bDebug=settings.bDebug)
                #actionSamples.append((data.T, data.SonT, data.is_valid))
                labelLow = 'below'
                labelHigh = getSignLabel()

            if labelHigh == 'unknown':
                if self.bAllowErrorsForTn:
                    # If we have sampled the action close to Tmax and saw it was below the target value, claim that the
                    # failed action sample even closer to Tmax would also be below. If we have not seen the action below
                    # the target value close to Tmax, claim that the action is instead above the target value.
                    labelHigh = 'below' if seenBelowAtTHigh else 'above'
                # else we would have returned already.

            if labelHigh == 'below':
                saveCurveShapeAnalysisData(True)
                return actionCurveShapeAnalysisData
        else:
            labelHigh = 'above'

        labelMid = 'unknown'
        TMid = 0.5*(TLow + THigh)

        # If both endpoints of the sample region are above the target value, bisect until we have one endpoint below the
        # target value.
        while labelLow == 'above' and labelHigh == 'above':
            lowerTSample = data.copy()
            data.T = TMid
            bSuccess = evaluateAction_internal()

            if not bSuccess:
                saveCurveShapeAnalysisData(False)
                return actionCurveShapeAnalysisData

            Tstep *= 0.5
            labelMid = getSignLabel()

            if labelMid == 'unknown':
                if self.bAllowErrorsForTn:
                    labelMid = 'below'
                # else we would have returned already.

            if labelMid == 'below' or labelMid == 'equal':
                TLow = TMid
                labelLow = labelMid
                break

            # If the midpoint is also above the target value, we need to determine whether the action is increasing or
            # decreasing here. This determines which half of the region the target value could lie within, as we are
            # seeking regions of lower action.

            # The temperature stored in data might be slightly different that TMid if there was a failed action calculation
            # and a later attempt was successful.
            midT = data.T
            midSonT = data.SonT
            lowerTSample = data.copy()
            data.T += Tstep
            bSuccess = evaluateAction_internal()

            if not bSuccess:
                saveCurveShapeAnalysisData(False)
                return actionCurveShapeAnalysisData

            if not data.is_valid:
                if self.bAllowErrorsForTn:
                    TLow = TMid
                # else we would have returned already.
            elif data.SonT > midSonT:
                THigh = TMid
            else:
                TLow = TMid

            # If the temperature window is becoming too small, claim that the target value cannot be found. The below
            # extrapolation step should make this unnecessary. TODO: [2023] why are we still doing it then?
            if (THigh - TLow) / (self.Tmax - self.Tmin) < 0.01:
                if self.bDebug:
                    print('Unable to find action below desired value before temperature window reduced to', THigh - TLow,
                        'GeV.')

                saveCurveShapeAnalysisData()
                return actionCurveShapeAnalysisData

            # Given the gradient of the action we have approximated, predict the temperature at which the target value
            # would be reached based on linear extrapolation. Since the gradient will overestimate the rate at which the
            # target value is reached (based on the possible shape of the action curve), if this predicted temperature lies
            # outside of the current temperature window, we can conservatively claim that the target value cannot be found.
            dTdS = (data.T - midT) / (data.SonT - midSonT)

            if data.SonT > midSonT:
                TPredicted = midT - (midSonT - self.actionSampler.maxSonTThreshold)*dTdS
                extrapolationFailed = TPredicted < TLow
            else:
                TPredicted = data.T - (data.SonT - self.actionSampler.maxSonTThreshold)*dTdS
                extrapolationFailed = TPredicted > THigh

            if extrapolationFailed:
                if self.bDebug:
                    print('S/T will not go below', self.actionSampler.maxSonTThreshold, 'based on linear extrapolation.')

                saveCurveShapeAnalysisData()
                return actionCurveShapeAnalysisData

            TMid = 0.5*(TLow + THigh)

        # Now we should have labelLow='below' and labelHigh='above'. That is, we bracket the desired solution. We can simply
        # bisect from here.
        while THigh - TLow > (self.Tmax - self.Tmin)*1e-5:
            lowerTSample = data.copy()
            TMid = 0.5*(TLow + THigh)
            data.T = TMid
            # Negative because thin-walled errors for T ~ Tc are more likely.
            Tstep = -0.05*(THigh - TLow)
            bSuccess = evaluateAction_internal()

            if not bSuccess:
                saveCurveShapeAnalysisData(False)
                return actionCurveShapeAnalysisData

            prevLabelMid = labelMid
            labelMid = getSignLabel()

            if labelMid == 'unknown':
                if self.bAllowErrorsForTn:
                    if prevLabelMid == 'below' or prevLabelMid == 'unknown':
                        THigh = TMid
                    else:
                        TLow = TMid
                # else we would have returned already.
            elif labelMid == 'equal':
                saveCurveShapeAnalysisData(True)
                return actionCurveShapeAnalysisData
            elif labelMid == 'above':
                THigh = TMid
            else:
                TLow = TMid

        saveCurveShapeAnalysisData(True)
        return actionCurveShapeAnalysisData

    def calculateGamma(self, T: Union[float, np.ndarray], action: Union[float, np.ndarray])\
            -> Union[float, np.ndarray]:
        return T**4 * (action/(2*np.pi))**(3/2) * np.exp(-action)

    # TODO: [2023] need to allow this to be used for list inputs again.
    #def calculateInstantaneousNucleationRate(T: Union[float, Iterable[float]], SonT: Union[float, Iterable[float]], potential: AnalysablePotential):
    def calculateInstantaneousNucleationRate(self, T: float, action: float) -> float:
        HSq = self.hubble_squared(T)

        Gamma = self.calculateGamma(T, action)

        return Gamma / (T*HSq**2)

    # TODO: We can optimise this for a list of input temperatures by reusing potential samples in adjacent derivatives.
    def hubble_squared(self, T: float) -> float:
        rhof = hydrodynamics.energy_density_from_phase(self.fromPhase, self.toPhase, self.potential, T)
        return hubble_squared_from_energy_density(rhof - self.groud_state_energy_density)

    def getHydroVars(self, T: float) -> HydroVars:
        return hydrodynamics.make_hydro_vars(self.fromPhase, self.toPhase, self.potential, T,
            self.groud_state_energy_density)

    def reheat_temperature(self, T: float) -> float:
        Tsep = min(0.001*(self.properties.Tc - self.Tmin), 0.5*(T - self.Tmin))
        rhof = hydrodynamics.energy_density_from_phase(self.fromPhase, self.toPhase, self.potential, T)
        def objective(t):
            rhot = hydrodynamics.energy_density_to_phase(self.fromPhase, self.toPhase, self.potential, t)
            # Conservation of energy => rhof = rhof*Pf + rhot*Pt which is equivalent to rhof = rhot (evaluated at
            # different temperatures, T and Tt (Treh), respectively).
            return rhot - rhof
              
        if objective(self.properties.Tc) >= 0:
            max_t = self.properties.Tc
        else:
            # If the energy density of the true vacuum is never larger than the current energy density of the false vacuum even
            # at Tc, then reheating goes beyond Tc
            max_t = self.toPhase.T[-1] - 2. * Tsep
            # Also, check the energy density of the true vacuum when it first appears. If a solution still doesn't exist
            # here, return None
            if max_t > self.properties.Tc and objective(max_t) < 0:
                warnings.warn("Cannot find reheat temperature")
                return None

        return scipy.optimize.toms748(objective, T, max_t)

    def transitionCouldComplete(self, maxAction: float, Pf: list[float]) -> bool:
        if self.actionSampler.T[-1] <= self.potential.minimum_temperature:
            return False

        # If the nucleation rate is still high.
        if self.actionSampler.SonT[-1] < maxAction:
            return True

        # Check if the transition progress is speeding up (i.e. the rate of change of Pf is increasing).
        if Pf[-2] - Pf[-1] > Pf[-3] - Pf[-2]:
            return True

        # If the transition has stagnated (or hasn't even begun).
        if Pf[-1] == Pf[-2]:
            return False

        # Assume the transition progress (i.e. change in Pf) is linear from here. Extrapolate to Tmin. Predict what
        # temperature would yield P(T) = 0.
        T0 = self.actionSampler.subT[-1] - Pf[-1] * (self.actionSampler.subT[-2] - self.actionSampler.subT[-1])\
             / (Pf[-2] - Pf[-1])

        # If this temperature is above the minimum temperature for which this transition can still occur, then it's possible
        # that it could complete.
        return T0 >= self.Tmin


def hubble_squared(fromPhase: Phase, toPhase: Phase, potential: AnalysablePotential, T: float,
                               groud_state_energy_density: float) -> float:
    rhof = hydrodynamics.energy_density_from_phase(fromPhase, toPhase, potential, T)
    return hubble_squared_from_energy_density(rhof - groud_state_energy_density)


def hubble_squared_from_energy_density(energy_density):
    return 8 * np.pi * GRAV_CONST / 3 * energy_density


# Returns T, SonT, bFinishedAnalysis.
def loadPrecomputedActionData(fileName: str, transition: Transition, maxSonTThreshold: float) -> tuple[list[float],
        list[float], bool]:
    if fileName == '':
        return [], [], False

    precomputedT = []
    precomputedSonT = []

    if fileName[-5:] == '.json':
        with open(fileName) as f:
            data = json.load(f)

            transDict = None
            for tr in data['transitions']:
                if tr['id'] == transition.ID:
                    transDict = tr
                    break

        if transDict is not None:
            precomputedT = transDict['T']
            precomputedSonT = transDict['SonT']

            # If the nucleation window wasn't found, we do not have precomputed data.
            if not transDict['foundNucleationWindow']:
                transition.analysis.T = precomputedT
                transition.analysis.SonT = precomputedSonT
                return [], [], True
        else:
            print('Unable to find transition with id =', transition.ID, 'in the JSON file.')
    elif fileName[-4:] == '.txt':
        data = np.loadtxt(fileName)

        if data is not None:
            precomputedT = data[..., 0][::-1]
            precomputedSonT = data[..., 1][::-1]
    else:
        print('Unsupported file extension', fileName.split('.')[-1], 'for precomputed action curve file.')

    if len(precomputedT) == 0:
        return [], [], False

    if len(precomputedSonT) > 0:
        if min(precomputedSonT) > maxSonTThreshold:
            transition.analysis.T = precomputedT
            transition.analysis.SonT = precomputedSonT
            transition.analysis.actionCurveFile = fileName
            return [], [], True

    return precomputedT, precomputedSonT, False
