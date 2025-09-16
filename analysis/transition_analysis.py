"""
Transiton analysis
=====================
"""

from __future__ import annotations

import time
import json
import traceback
from typing import Optional, Union, List
import math

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

from gws import hydrodynamics
from gws.hydrodynamics import HydroVars
from util.print_suppressor import PrintSuppressor
from util import integration
from util.events import notifyHandler
from models.analysable_potential import AnalysablePotential
from analysis.phase_structure import Phase, Transition
from util.integration import LinearNestedNormalisedIntegrationHelper, CubedNestedIntegrationHelper


totalActionEvaluations = 0

# TODO: should move to a constants file.
GRAV_CONST = 6.7088e-39


# TODO: not currently thrown anywhere. Replace fail flags and messages with exceptions.
class FailedActionCalculationException(Exception):
    pass


class ActionSample:
    def __init__(self, T: float, S3: float = -1.):
        self.T = T
        self.S3 = S3
        # TODO: doesn't correctly handle low-temperature transitions T; e.g. supercooling down to (sub-)MeV scale.
        self.action = -1 if S3 < 0. else S3 / max(0.001, T)
        self.bValid = self.S3 > 0.

    # Emulates a constructor that takes as input another TemperatureData object.
    # See https://stackoverflow.com/questions/141545/how-to-overload-init-method-based-on-argument-type.
    @classmethod
    def copyData(cls, data: 'ActionSample') -> 'ActionSample':
        return cls(data.T, data.S3)

    # Need to use quotes around type annotation since it's within the same class.
    # See https://stackoverflow.com/a/49392996.
    def transferData(self, data: 'ActionSample'):
        data.T = self.T
        data.S3 = self.S3
        data.action = self.action
        data.bValid = self.bValid

    def __str__(self):
        return f'(T: {self.T}, S/T: {self.action})'

    def __repr__(self):
        return str(self)


# TODO: either move this all to Transition, or move all analysis quantities from Transition to here.
class AnalysedTransition:
    def __init__(self):
        self.actionTn = -1
        self.actionTnbar = -1
        self.actionTp = -1
        self.actionTe = -1
        self.actionTf = -1

        self.betaTn = -1
        self.betaTnbar = -1
        self.betaTp = -1
        self.betaTe = -1
        self.betaTf = -1
        self.betaV = -1

        self.Hn = -1
        self.Hnbar = -1
        self.Hp = -1
        self.He = -1
        self.Hf = -1

        self.T = []
        self.action = []
        self.lowestAction = -1
        self.error = ''
        self.actionCurveFile = ''


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

    # precomputedT and precomputedAction are lists of values for the S/T curve. These can be used to avoid sampling the
    # action curve explicitly, until the samples run out.
    def __init__(self, transitionAnalyser: 'TransitionAnalyser', minActionThreshold: float, maxActionThreshold: float,
                 actionTargetTolerance: float, stepSizeMax=0.95, precomputedT: Optional[list[float]] = None,
                 precomputedAction: Optional[list[float]] = None):
        self.transitionAnalyser = transitionAnalyser

        # Copy properties for concision.
        self.bDebug = transitionAnalyser.bDebug
        self.potential = transitionAnalyser.potential
        self.fromPhase = transitionAnalyser.fromPhase
        self.toPhase = transitionAnalyser.toPhase
        self.Tmin = transitionAnalyser.Tmin
        self.Tmax = transitionAnalyser.Tmax

        self.precomputedT = precomputedT if precomputedT is not None else []
        self.precomputedAction = precomputedAction if precomputedAction is not None else []

        if len(self.precomputedT) != len(self.precomputedAction):
            self.precomputedT = []
            self.precomputedAction = []
            self.bUsePrecomputedSamples = False
        else:
            self.bUsePrecomputedSamples = len(self.precomputedT) > 0

        self.T = []
        self.action = []

        self.subT = []
        self.subAction = []
        self.subRhoV = []
        self.subRhof = []
        self.subRhot = []

        self.lowerActionData = []

        self.stepSize = -1
        self.stepSizeMax = stepSizeMax

        self.minActionThreshold = minActionThreshold
        self.maxActionThreshold = maxActionThreshold
        self.actionTargetTolerance = actionTargetTolerance

        notifyHandler.handleEvent(self, 'on_create')

    def calculateStepSize(self, low: ActionSample = None, mid: ActionSample = None, high: ActionSample = None):
        lowT = self.T[-1] if low is None else low.T
        midT = self.T[-2] if mid is None else mid.T
        highT = self.T[-3] if high is None else high.T
        lowAction = self.action[-1] if low is None else low.action
        midAction = self.action[-2] if mid is None else mid.action
        highAction = self.action[-3] if high is None else high.action

        # Check for linearity. Create a line between the lowest and highest temperature points, and check how far from
        # the average value the middle temperature point is.
        interpValue = lowAction + (highAction - lowAction) * (midT - lowT) / (highT - lowT)
        linearity = 1 - 2*abs(midAction - interpValue) / abs(highAction - lowAction)

        if self.stepSize == -1:
            self.stepSize = min(self.stepSizeMax, 1 - (abs(midAction - lowAction) / midAction))

        # This gives (lin, stepFactor): (0.99, 1.0315), (0.98, 0.9186), (0.94, 0.8) and 0.5 for lin <= 0.8855.
        stepFactor = max(0.5, 0.8 + 0.4*((linearity - 0.94)/(1 - 0.94))**3)
        self.stepSize = min(self.stepSizeMax, 1 - stepFactor*(1 - self.stepSize))

        if self.bDebug:
            print('size:', self.stepSize, 'factor:', stepFactor, 'lin:', linearity)

    def getNextSample_new(self, sampleData: ActionSample) -> tuple[bool, str]:
        print('=======================================================================================================')
        print('getNextSample_new')
        print('=======================================================================================================')
        # If we are already near the minimum temperature allowed for phase transitions in this potential, we assume the
        # transition will not progress from here. Or, if it does, we cannot accurately determine its progress due to
        # external effects (like other cosmological events) that we don't handle.
        if len(self.T) > 0 and sampleData.T <= self.potential.minimumTemperature:
            return False, 'Reached minimum temperature'

        if len(self.T) >= 3:
            actionInterp: np.poly1d = lagrange(self.T[-3:], self.action[-3:])
        else:
            actionInterp: np.poly1d = lagrange(self.T[-2:], self.action[-2:])

        temperatureStepSize = self.T[-2] - self.T[-1]
        minStepSize: float
        maxStepSize: float

        minStepSize, maxStepSize = self.findTemperatureForFalseVacuumFraction(
            T_prev=self.T[-1],
            delta_Pf_min=self.transitionAnalyser.falseVacuumFractionChange_samples_min,
            delta_Pf_max=self.transitionAnalyser.falseVacuumFractionChange_samples_max,
            initial_step_size=temperatureStepSize,
            action_interp=actionInterp,
            T_low=max(1., self.Tmin), # TODO: undo this temporary hack
            T_high=self.T[-1]
        )

        # TODO: for now, just pick the smaller step size.
        temperatureStepSize = minStepSize

        sampleData.T = self.T[-1] - temperatureStepSize
        self.evaluateAction(sampleData)

        if not sampleData.bValid:
            if self.bDebug:
                print('Failed to evaluate action at trial temperature T=', sampleData.T)
            return False, 'Action failed'

        self.T.append(sampleData.T)
        self.action.append(sampleData.action)

        return True, 'Success'

    # TODO: need type for action interp.
    def findTemperatureForFalseVacuumFraction(self, T_prev: float, delta_Pf_min: float, delta_Pf_max: float,
                                              initial_step_size: float, action_interp, T_low: float, T_high: float)\
            -> tuple[float, float]:
        print(f'Finding temperature for Pf within window: [{T_low}, {T_high}]')
        falseVacuumFractionPrevious: float = self.transitionAnalyser.falseVacuumFraction[-1]
        temperatureStepSize: float = initial_step_size

        # Whether the previous predicted false vacuum fraction change is too small or too large.
        wasTooLarge: bool = False
        wasTooSmall: bool = False

        # The largest sampled step size for which the change in false vacuum fraction was too small.
        minStepSize: float = 0.
        # The smallest sampled step size for which the change in false vacuum fraction was too large.
        maxStepSize: float = T_high - T_low

        if action_interp(T_low) > 0:
            falseVacuumFraction: float = self.predictFalseVacuumFraction(T_prev, maxStepSize, action_interp,
                                                                         calculate_midpoint=True)

            falseVacuumFractionChange: float = abs(falseVacuumFraction - falseVacuumFractionPrevious)

            if falseVacuumFractionChange < delta_Pf_min:
                print('Predicting Pf will not change enough across the temperature window.')
                return maxStepSize, maxStepSize
        else:
            print('Unable to check Pf at max temperature step due to negative action extrapolation.')

        # Adjust the temperature step size until we can bracket the desired range.
        while True:
            falseVacuumFraction = self.predictFalseVacuumFraction(T_prev, temperatureStepSize, action_interp,
                                                                  calculate_midpoint=True)

            falseVacuumFractionChange = abs(falseVacuumFraction - falseVacuumFractionPrevious)

            tooSmall = falseVacuumFractionChange < delta_Pf_min
            tooLarge = falseVacuumFractionChange > delta_Pf_max

            if tooSmall:
                print(f'Too small: Pf change = {falseVacuumFractionChange:.3e} not in [{delta_Pf_min:.3e}, '
                      f'{delta_Pf_max:.3e}]')
                if wasTooLarge:
                    # We know the correct step size is somewhere in the range temperatureStepSize*(1, 2).
                    minStepSize = temperatureStepSize
                    maxStepSize = temperatureStepSize*2
                    break
                else:
                    wasTooSmall = True
                    minStepSize = temperatureStepSize
                    # if temperatureStepSize == maximumStepSize:
                    #    print('Already tried the maximum step size and it yielded insufficient change in false vacuum'
                    #          ' fraction.')
                    #    return max(2, math.ceil(temperatureChange_samples / temperatureStepSize))
                    if temperatureStepSize >= (T_high - T_low):
                        print('Temperature step size exceeds temperature window.')
                        return maxStepSize, maxStepSize
                    temperatureStepSize *= 2
            elif tooLarge:
                print(f'Too large: Pf change = {falseVacuumFractionChange:.3e} not in [{delta_Pf_min:.3e}, '
                      f'{delta_Pf_max:.3e}]')
                if wasTooSmall:
                    # We know the correct step size is somewhere in the range temperatureStepSize*(0.5, 1).
                    minStepSize = 0.5*temperatureStepSize
                    maxStepSize = temperatureStepSize
                    break
                else:
                    wasTooLarge = True
                    temperatureStepSize *= 0.5
            else:
                break

        return minStepSize, maxStepSize

    # TODO: need type for actionInterp.
    def predictFalseVacuumFraction(self, prevT: float, temperatureStepSize: float, actionInterp,
                                   calculate_midpoint: bool) -> float:
        print('Predicting Pf for T:', prevT - temperatureStepSize, 'where action should be:',
              actionInterp(prevT - temperatureStepSize))

        # The last subsample that has already been integrated.
        lastT: float = self.subT[-1]

        self.transitionAnalyser.integrationHelper_trueVacVol.setRestorePoint()
        self.transitionAnalyser.integrationHelper_avgBubRad.setRestorePoint()

        # TODO: update documentation with findings about necessity of midpoint.
        # First estimate Pf at the midpoint, halfway between the previous temperature and the trial temperature.
        # This is necessary (at least initially) to estimate the change in Pf. There are two effects that change Pf:
        # 1. Growth of existing bubbles.
        # 2. Nucleation of new bubbles.
        # Only the growth is captured when adding another step in the integration for Pf. Newly nucleated bubbles
        # likely have negligible initial radius so will not affect Pf at the temperature at which they were
        # nucleated.
        # Sampling at the midpoint is especially important when len(self.T) == 2, because otherwise the change in
        # Pf will only come from the nucleation of new bubbles that have no time to have grown.
        if calculate_midpoint:
            # TODO: is it always necessary to sample Pf at the midpoint? It will give a better estimate of Pf at the
            #  trial temperature but comes at twice the cost. Provide an option and test effects.
            T: float = prevT - 0.5*temperatureStepSize
            self.subT.append(T)
            action: float = actionInterp(T)
            self.subAction.append(action)
            hydroVars: HydroVars = self.transitionAnalyser.getHydroVars(T)
            self.transitionAnalyser.integrateTemperatureStep(T, action, lastT, hydroVars)

        # Now estimate Pf at the trial temperature.
        T: float = prevT - temperatureStepSize
        self.subT.append(T)
        action: float = actionInterp(T)
        self.subAction.append(action)
        hydroVars: HydroVars = self.transitionAnalyser.getHydroVars(T)
        self.transitionAnalyser.integrateTemperatureStep(T, action, lastT, hydroVars)

        Pf: float = self.transitionAnalyser.falseVacuumFraction[-1]

        # Undo the integration (which involved many list appends) because we only wanted to estimate Pf, not store
        # this new data point.
        #self.transitionAnalyser.undoIntegration(2)
        self.subT.pop()
        self.subAction.pop()

        if calculate_midpoint:
            #self.transitionAnalyser.undoIntegration()
            self.subT.pop()
            self.subAction.pop()

        self.transitionAnalyser.undoIntegration(2 if calculate_midpoint else 1)

        return Pf

    # TODO: old version used before 14/09/2024
    def getNextSample(self, sampleData: ActionSample, Gamma: list[float], numBubbles: float, Tmin: float)\
            -> (bool, str):
        # If we are already near T=0, the transition is assumed to not progress from here. We consider only bubble
        # nucleation via thermal fluctuations.
        #if len(self.T) > 0 and sampleData.T <= 0.001:
        #    return False, 'Freeze out'

        # If we are already near the minimum temperature allowed for phase transitions in this potential, we assume the
        # transition will not progress from here. Or, if it does, we cannot accurately determine its progress due to
        # external effects (like other cosmological events) that we don't handle.
        if len(self.T) > 0 and sampleData.T <= self.potential.minimumTemperature:
            return False, 'Reached minimum temperature'

        # Remove all stored data points whose temperature is larger than the last sampled temperature.
        while len(self.lowerActionData) > 0 and self.lowerActionData[-1].T >= self.T[-1]:
            self.lowerActionData.pop()

        if self.bUsePrecomputedSamples:
            if len(self.T) < len(self.precomputedT):
                self.T.append(self.precomputedT[len(self.T)])
                self.action.append(self.precomputedAction[len(self.action)])
                sampleData.T = self.T[-1]
                sampleData.action = self.action[-1]
                return True, 'Precomputed'
            elif len(self.T) == len(self.precomputedT):
                # TODO: not exactly sure if this is necessary, but probably is.
                self.calculateStepSize()

        # Construct a quadratic Lagrange interpolant from the three most recent action samples.
        quadInterp = lagrange(self.T[-3:], self.action[-3:])
        # Extrapolate with the same step size as between the last two samples.
        TNew = max(self.Tmin*1.001, 2*self.T[-1] - self.T[-2])
        actionNew = quadInterp(TNew)

        # If we are sampling the same point because we've reached Tmin, then the transition cannot progress any
        # further.
        if self.T[-1] == self.Tmin*1.001:
            if self.bDebug:
                print('Already sampled near Tmin =', sampleData.T, '-- transition analysis halted.')
            return False, 'Reached Tmin'

        # Determine the nucleation rate for nearby temperatures under the assumption that quadratic extrapolation is
        # appropriate.

        GammaNew = self.transitionAnalyser.calculateBubbleNucleationRate(TNew, actionNew)
        GammaCur = self.transitionAnalyser.calculateBubbleNucleationRate(self.T[-1], self.action[-1])
        GammaPrev = self.transitionAnalyser.calculateBubbleNucleationRate(self.T[-2], self.action[-2])

        def nearMaxNucleation() -> bool:
            dSdTnew = (self.action[-1] - actionNew) / (self.T[-1] - TNew)
            dSdT = (self.action[-2] - self.action[-1]) / (self.T[-2] - self.T[-1])
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
                    extraBubbles = self.calculateExtraBubbles(TNew, actionNew, Tmin)

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
                    extraBubbles = self.calculateExtraBubbles(TNew, actionNew, Tmin)

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
                    extraBubbles = self.calculateExtraBubbles(TNew, actionNew, Tmin)

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
                        extraBubbles = self.calculateExtraBubbles(TNew, actionNew, Tmin)

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
        if numBubbles < 1 and actionNew < 160:
            if stepFactor > 1:
                stepFactor = 1 + 0.85*(stepFactor - 1)
            else:
                stepFactor *= 0.85

        TNew = max(self.Tmin*1.001, self.T[-1] - max(stepFactor*(self.T[-2] - self.T[-1]),
            self.potential.minimumTemperature))

        # Prevent large steps near T=Tmin causing large errors in the interpolated action from affecting ultracooled
        # transitions.
        if (TNew - self.Tmin) < 0.5*(self.T[-1] - self.Tmin) and (self.T[-1] - self.Tmin) > 5.:
            TNew = 0.5*(self.Tmin + self.T[-1])

        # Hack for better resolution in supercool GW plots.
        #maxStep = 0.1 # BP1 and BP2
        #maxStep = 0.3 # BP3 and BP4

        #if abs(Tnew - self.T[-1]) > maxStep:
        #    Tnew = self.T[-1] - maxStep

        sampleData.T = TNew

        self.evaluateAction(sampleData)

        if not sampleData.bValid:
            if self.bDebug:
                print('Failed to evaluate action at trial temperature T=', sampleData.T)
            return False, 'Action failed'

        self.T.append(sampleData.T)
        self.action.append(sampleData.action)

        return True, 'Success'

    def calculateExtraBubbles(self, Tnew: float, actionNew: float, Tmin: float) -> float:
        extraBubbles = 0
        numPoints = 20

        TList = np.linspace(self.T[-1], Tnew, numPoints)
        # TODO: replace with quadratic interpolation.
        actionList = np.linspace(self.action[-1], actionNew, numPoints)
        GammaList = self.transitionAnalyser.calculateBubbleNucleationRate(TList, actionList)
        energyStart = hydrodynamics.calculateEnergyDensityAtT_singlePhase(self.fromPhase, self.toPhase, self.potential,
            TList[0], forFromPhase=True) - self.transitionAnalyser.groundStateEnergyDensity
        energyEnd = hydrodynamics.calculateEnergyDensityAtT_singlePhase(self.fromPhase, self.toPhase, self.potential,
            TList[-1], forFromPhase=True) - self.transitionAnalyser.groundStateEnergyDensity
        # TODO: replace with quadratic interpolation.
        energyDensityList = np.linspace(energyStart, energyEnd, numPoints)
        #HList = self.transitionAnalyser.calculateHubbleParameterSq_supplied(energyDensityList)
        HList = [calculateHubbleParameterSq_supplied(e) for e in energyDensityList]
        integrandList = [GammaList[i]/(TList[i]*HList[i]**2) for i in range(numPoints)]

        for i in range(1, numPoints):
            extraBubbles += 0.5*(integrandList[i] + integrandList[i-1]) * (TList[i-1] - TList[i])

        return extraBubbles

    # TODO: old version used before 21/05/2024
    def getNumSubsamples_old(self):
        # Interpolate between the last two sample points, creating a densely sampled line which we can integrate.
        # Increase the sampling density as the action decreases due to the exponentially larger nucleation probability.
        quickFactor = 1.
        # return 3 #max(5, int(100*(self.T[-2] - self.T[-1])))
        #numSamples = int(quickFactor*(100 + 1000*))
        # TODO: make the number of interpolation samples have some physical motivation (e.g. ensure that the true
        #  vacuum fraction can change by no more than 0.1% of its current value across each sample).
        actionFactor = abs(self.action[-2] / self.action[-1] - 1) / (1 - self.minActionThreshold / self.maxActionThreshold)
        dTFactor = abs(min(2,self.T[-2]/self.T[-1]) - 1) / (1 - self.Tmin/self.T[0])
        numSamples = max(1, int(quickFactor*(2 + np.sqrt(1000*(actionFactor + dTFactor)))))
        if self.bDebug:
            print('Num samples:', numSamples, actionFactor, dTFactor)
        return numSamples

    def getNumSubsamples(self) -> int:
        print('=======================================================================================================')
        print('getNumSubsamples')
        print('=======================================================================================================')

        # The trial temperature should depend on the ratio of the desired change in Pf between subsamples, and the
        # desired change in Pf between the last two samples. Assume that Pf will change at an approximately linear
        # rate throughout the sample window.
        falseVacuumFractionChange_samples: float = 0.5*(self.transitionAnalyser.falseVacuumFractionChange_samples_min
            + self.transitionAnalyser.falseVacuumFractionChange_samples_max)
        falseVacuumFractionChange_subsamples: float =\
            0.5*(self.transitionAnalyser.falseVacuumFractionChange_subsamples_min
            + self.transitionAnalyser.falseVacuumFractionChange_subsamples_max)
        temperatureChange_samples: float = self.T[-2] - self.T[-1]
        temperatureStepSize: float = falseVacuumFractionChange_subsamples / falseVacuumFractionChange_samples\
            * temperatureChange_samples

        if len(self.T) >= 3:
            actionInterp: np.poly1d = lagrange(self.T[-3:], self.action[-3:])
        else:
            actionInterp: np.poly1d = lagrange(self.T[-2:], self.action[-2:])

        # The last subsample that has already been integrated.
        lastT: float = self.subT[-1]

        falseVacuumFractionPrevious: float = self.transitionAnalyser.falseVacuumFraction[-1]
        falseVacuumFractionChange: float
        falseVacuumFraction: float

        # Whether the current predicted false vacuum fraction change is too small or too large.
        tooSmall: bool
        tooLarge: bool

        # The largest sampled step size for which the change in false vacuum fraction was too small.
        minStepSize: float
        # The smallest sampled step size for which the change in false vacuum fraction was too large.
        maxStepSize: float
        # Collectively, these are used to bracket the range of step sizes that could lead to the desired change in false
        # vacuum fraction.

        #temperatureForZeroAction = 0.

        #if len(actionInterp.roots) > 0:
        #    if np.isreal(actionInterp.roots[0]):
        #        temperatureForZeroAction = max(0, actionInterp.roots[0])

        #minimumTemperature = temperatureForZeroAction + 0.01*(lastT - temperatureForZeroAction)
        #maximumStepSize = lastT - minimumTemperature

        def num_subsamples(step_size: float) -> int:
            # We need at least two samples.
            return max(2, math.ceil(temperatureChange_samples / step_size))

        minStepSize, maxStepSize = self.findTemperatureForFalseVacuumFraction(
            T_prev=self.T[-2],
            delta_Pf_min=self.transitionAnalyser.falseVacuumFractionChange_subsamples_min,
            delta_Pf_max=self.transitionAnalyser.falseVacuumFractionChange_subsamples_max,
            initial_step_size=temperatureStepSize,
            action_interp=actionInterp,
            T_low=self.T[-1],
            T_high=self.T[-2]
        )

        # Now bisect between minStepSize and maxStepSize. The backup termination condition is that the bisection window
        # has become small enough to not affect the result. Because we seek an integer number of subsamples, two float
        # step sizes can lead to the same integer result. There is no need to bisect further if this point is reached.
        # There are edge cases where the min and max step sizes straddle an integer division solution
        while num_subsamples(maxStepSize) - num_subsamples(minStepSize) > 1:
            temperatureStepSize = 0.5*(minStepSize + maxStepSize)
            falseVacuumFraction = self.predictFalseVacuumFraction(self.T[-2], temperatureStepSize, actionInterp,
                                                                  calculate_midpoint=True)
            falseVacuumFractionChange = falseVacuumFraction - falseVacuumFractionPrevious

            tooSmall = falseVacuumFractionChange < self.transitionAnalyser.falseVacuumFractionChange_subsamples_min
            tooLarge = falseVacuumFractionChange > self.transitionAnalyser.falseVacuumFractionChange_subsamples_max

            if tooSmall:
                minStepSize = temperatureStepSize
            elif tooLarge:
                maxStepSize = temperatureStepSize
            else:
                break

        return num_subsamples(temperatureStepSize)

    # Stores newData in lowerActionData, maintaining the sorted order.
    def storeLowerActionData(self, newData):
        if len(self.lowerActionData) == 0 or self.lowerActionData[-1].T <= newData.T:
            self.lowerActionData.append(newData)
        else:
            i = 0

            while i < len(self.lowerActionData) and newData.T > self.lowerActionData[i].T:
                i += 1

            self.lowerActionData.insert(i, newData)

    # Throws ThinWallError from CosmoTransitions.
    # Throws unhandled exceptions from CosmoTransitions.
    # Throws unhandled InvalidTemperatureException from findPhaseAtT.
    def evaluateAction(self, data: ActionSample) -> tuple[np.ndarray, np.ndarray]:
        # Do optimisation.
        T = data.T
        fromFieldConfig = self.fromPhase.findPhaseAtT(T, self.potential)
        toFieldConfig = self.toPhase.findPhaseAtT(T, self.potential)

        # TODO: make factor configurable.
        fieldSeparationScale = 0.001*self.potential.fieldScale

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
        data.bValid = False
        startTime = time.perf_counter()

        if self.bDebug:
            print('Evaluating action at T =', T)

        def V(X): return self.potential.Vtot(X, T)
        def gradV(X): return self.potential.gradV(X, T)

        global totalActionEvaluations
        totalActionEvaluations += 1

        if not self.bVerbose:
            with PrintSuppressor():
                action = pathDeformation.fullTunneling([toFieldConfig, fromFieldConfig], V, gradV,
                    V_spline_samples=self.numSplineSamples, maxiter=self.maxIter,
                    verbose=self.bVerbose, tunneling_findProfile_params=tunneling_findProfile_params).action
        else:
            action = pathDeformation.fullTunneling([toFieldConfig, fromFieldConfig], V, gradV,
                V_spline_samples=self.numSplineSamples, maxiter=self.maxIter,
                verbose=self.bVerbose, tunneling_findProfile_params=tunneling_findProfile_params).action

        # Update the data, including changes to the phases and temperature if the calculation previously failed.
        data.T = T
        data.S3 = action
        # TODO: we should probably handle this better.
        # At T=0, emulate the freezing suppression by dividing by a small but non-vanishing number. In case the action
        # calculation returns a negative result (probably due to a failure of the algorithm), assume the action is actually
        # zero.
        data.action = max(0, action / max(T, 0.001))
        data.bValid = data.action > 1e-10

        if self.bDebug:
            if data.bValid:
                print(f'Successfully evaluated action S = {data.action} in {time.perf_counter() - startTime} seconds.')
            else:
                print(f'Obtained nonsensical action S = {data.action} in {time.perf_counter() - startTime} seconds.')

        return fromFieldConfig, toFieldConfig


class ActionCurveShapeAnalysisData:
    def __init__(self):
        self.desiredData: Optional[ActionSample] = None
        self.nextBelowDesiredData: Optional[ActionSample] = None
        self.storedLowerActionData: List[ActionSample] = []
        self.actionSamples: List[ActionSample] = []
        # TODO: does the default choice for this ever matter? Are there cases where we don't identify it but still
        #  continue the transition analysis? Maybe for fine-tuned subcritical transitions that nucleate quickly but
        #  don't complete quickly?
        self.bBarrierAtTmin: bool = False
        self.confidentNoNucleation: bool = False
        self.error: bool = False

    def copyDesiredData(self, sample: ActionSample):
        self.desiredData = ActionSample.copyData(sample)

    def copyNextBelowDesiredData(self, sample: ActionSample):
        self.nextBelowDesiredData = ActionSample.copyData(sample)

    def copyStoredLowerActionData(self, samples: List[ActionSample]):
        for sample in samples:
            self.storedLowerActionData.append(ActionSample.copyData(sample))

    def copyActionSamples(self, samples: List[ActionSample]):
        for sample in samples:
            # samples is a list of tuples of primitive types, so no need for a manual deep copy.
            if sample.bValid:
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
    groundStateEnergyDensity: float

    # The lowest temperature for which the transition is still possible. We don't want to check below this, as one
    # of the phases may have disappeared, their stability may reverse, or T=0 may have been reached.
    Tmin: float = 0.
    Tmax: float = 0.
    Tstep: float = 0.

    actionSampler: ActionSampler

    bComputeSubsampledThermalParams: bool

    # TODO: make vw a function of this class that can be overridden. Currently it is obtained from the transition.
    def __init__(self, potential: AnalysablePotential, transition: Transition, fromPhase: Phase, toPhase: Phase,
            groundStateEnergyDensity: float, Tmin: float = 0., Tmax: float = 0.):
        self.bubbleWallVelocity: List[float] = []
        self.meanBubbleSeparation: List[float] = []
        self.meanBubbleRadius: List[float] = []
        self.physicalVolume: List[float] = []
        self.hydroVars: List[HydroVars] = []
        self.hubbleParameter: List[float] = []
        self.bubbleNumberDensity: List[float] = []
        self.nucleationRate: List[float] = []
        self.falseVacuumFraction: List[float] = []
        self.trueVacuumVolumeExtended: List[float] = []
        self.numBubblesCorrectedIntegral: List[float] = []
        self.numBubblesIntegral: List[float] = []
        self.numBubblesCorrectedIntegrand: List[float] = []
        self.numBubblesIntegrand: List[float] = []
        self.beta: List[float] = []
        self.potential: AnalysablePotential = potential
        self.transition: Transition = transition
        self.fromPhase: Phase = fromPhase
        self.toPhase: Phase = toPhase
        self.groundStateEnergyDensity: float = groundStateEnergyDensity
        self.integrationHelper_trueVacVol: Optional[CubedNestedIntegrationHelper] = None
        self.integrationHelper_avgBubRad: Optional[LinearNestedNormalisedIntegrationHelper] = None

        self.Tmin: float = Tmin
        self.Tmax: float = Tmax

        if self.Tmin == 0:
            # The minimum temperature for which both phases exist, and prevent analysis below the effective potential's
            # cutoff temperature. Below this cutoff temperature, external effects may dramatically affect the phase
            # transition and cannot be captured here in a generic way.
            self.Tmin = max(self.fromPhase.T[0], self.toPhase.T[0], self.potential.minimumTemperature)

        if self.Tmax == 0:
            # The transition is not evaluated subcritically.
            self.Tmax = self.transition.Tc

        self.Tstep: float = max(0.0005*min(self.fromPhase.T[-1], self.toPhase.T[-1]),
                                0.0001*self.potential.temperatureScale)

        self.bComputeSubsampledThermalParams = False

        # ==============================================================================================================
        # Settings.
        # ==============================================================================================================

        self.percolationThreshold_Pf: float = 0.71
        self.percolationThreshold_Vext: float = -np.log(self.percolationThreshold_Pf)
        # When the fraction of space remaining in the false vacuum falls below this threshold, the transition is
        # considered to be complete.
        self.completionThreshold: float = 1e-2

        self.falseVacuumFractionChange_samples_min: float = 1e-2
        self.falseVacuumFractionChange_samples_max: float = 5e-2
        self.falseVacuumFractionChange_subsamples_min: float = 1e-3
        self.falseVacuumFractionChange_subsamples_max: float = 2e-3

        notifyHandler.handleEvent(self, 'on_create')

    def getBubbleWallVelocity(self, hydroVars: HydroVars) -> float:
        if self.bUseChapmanJouguetVelocity:
            thetaf = (hydroVars.energyDensityFalse - hydroVars.pressureFalse/hydroVars.soundSpeedSqTrue) / 4
            thetat = (hydroVars.energyDensityTrue - hydroVars.pressureTrue/hydroVars.soundSpeedSqTrue) / 4

            alpha = 4*(thetaf - thetat) / (3*hydroVars.enthalpyDensityFalse)

            cstSq = hydroVars.soundSpeedSqTrue
            cst = np.sqrt(cstSq)
            vw = (1 + np.sqrt(3*alpha*(1 - cstSq + 3*cstSq*alpha))) / (1/cst + 3*cst*alpha)
            print("thetaf = ", thetaf)
            print("thetat = ", thetat)
            print("alpha = ", alpha)
            print("cstSq = ", cstSq)
            print("cst = ", cst)
            print("hydroVars.energyDensityFalse = ", hydroVars.energyDensityFalse)
            print("hydroVars.energyDensityTrue = ", hydroVars.energyDensityTrue)
            print("hydroVars.pressureFalse= ", hydroVars.pressureFalse)
            print("hydroVars.pressureTrue= ", hydroVars.pressureTrue)
            print("hydroVars.soundSpeedSqTrue= ", hydroVars.soundSpeedSqTrue)
            print("hydroVars.soundSpeedSqFalse= ", hydroVars.soundSpeedSqFalse)
            
            if np.isnan(vw) or vw > 1.:
                print("Warning: finding vw = ", vw,  " adjusting to 1")
                #raise Exception("finding vw = ", vw)
                return 1.
            return vw
        else:
            return self.transition.vw

    # TODO: need to handle subcritical transitions better. Shouldn't use integration if the max sampled action is well
    #  below the nucleation threshold. Should treat the action as constant or linearise it and estimate transition
    #  temperatures under that approximation.
    def analyseTransition(self, startTime: float = -1.0, precomputedActionCurveFileName: str = ''):
        # TODO: this should depend on the scale of the transition, so make it configurable.
        # Estimate the maximum significant value of S/T by finding where the instantaneous nucleation rate multiplied by
        # the maximum possible duration of the transition is O(1). This is highly conservative, but intentionally so
        # because we only sample maxActionThreshold within some (loose) tolerance.
        maxActionThreshold = self.estimateMaximumSignificantAction() + 80
        # TODO: this also needs to be configurable or more general.
        minActionThreshold = 80.0
        toleranceAction = 3.0

        precomputedT, precomputedAction, bFinishedAnalysis = loadPrecomputedActionData(precomputedActionCurveFileName,
            self.transition, maxActionThreshold)

        if bFinishedAnalysis:
            return

        self.actionSampler = ActionSampler(self, minActionThreshold, maxActionThreshold, toleranceAction,
                                           precomputedT=precomputedT, precomputedAction=precomputedAction)

        if self.bDebug:
            print('Tmin:', self.Tmin)
            print('Tmax:', self.Tmax)
            print('Tc:', self.transition.Tc)
            print('bubbleWallVelocity:', self.transition.vw)

        if len(precomputedT) == 0:
            # TODO: we don't use allSamples anymore.
            sampleData, allSamples = self.primeTransitionAnalysis(startTime)

            if self.timeout_phaseHistoryAnalysis > 0:
                endTime = time.perf_counter()
                if endTime - startTime > self.timeout_phaseHistoryAnalysis:
                    return

            if sampleData is None:
                self.transition.bFoundNucleationWindow = False
                return

            self.transition.bFoundNucleationWindow = True

            # Remove any lowerActionData points that are very close together. We don't need to sample the action curve
            # extremely densely (a spacing of 1 is more than reasonable), and doing so causes problems with the
            # subsequent steps along the curve TODO: to be fixed anyway!
            if len(self.actionSampler.lowerActionData) > 0:
                keepIndices = [len(self.actionSampler.lowerActionData) - 1]
                for i in range(len(self.actionSampler.lowerActionData) - 2, -1, -1):
                    # Don't discard the point if it is separated in temperature from the almost degenerate action value
                    # already stored.
                    # TODO: make the action difference configurable. It should depend on the scale of the transition and
                    #  the desired precision. We should have some tunable 'minimum action sample difference' parameter
                    #  which is intelligently derived from the precision and new physics scale.
                    if abs(self.actionSampler.lowerActionData[i].action -
                           self.actionSampler.lowerActionData[keepIndices[-1]].action) > 1 or\
                            abs(self.actionSampler.lowerActionData[i].T -
                                self.actionSampler.lowerActionData[keepIndices[-1]].T) >\
                            self.potential.temperatureScale*0.001:
                        keepIndices.append(i)
                    elif self.bDebug:
                        print('Removing stored lower S/T data:', self.actionSampler.lowerActionData[i], 'because it is '
                            'too close to', self.actionSampler.lowerActionData[keepIndices[-1]])

                self.actionSampler.lowerActionData = [self.actionSampler.lowerActionData[i] for i in keepIndices]
        else:
            self.transition.bFoundNucleationWindow = True

            sampleData = ActionSample(-1, -1)
            #self.actionSampler.getNextSample(sampleData, [], 0., self.Tmin)
            #self.actionSampler.getNextSample(sampleData, [], 0., self.Tmin)
            self.actionSampler.getNextSample_new(sampleData)
            self.actionSampler.getNextSample_new(sampleData)

        self.hydroVars.append(self.getHydroVars(self.actionSampler.T[0]))
        self.bubbleWallVelocity.append(self.getBubbleWallVelocity(self.hydroVars[0]))
        self.meanBubbleSeparation.append(0.)
        self.meanBubbleRadius.append(0.)
        self.physicalVolume.append(3.)
        self.hubbleParameter.append(np.sqrt(self.calculateHubbleParameterSq_fromHydro(self.hydroVars[0])))
        self.bubbleNumberDensity.append(0.)
        self.nucleationRate.append(0.)
        self.falseVacuumFraction.append(1.)
        self.trueVacuumVolumeExtended.append(0.)
        self.numBubblesCorrectedIntegral.append(0.)
        self.numBubblesIntegral.append(0.)
        self.numBubblesCorrectedIntegrand.append(0.)
        self.numBubblesIntegrand.append(0.)
        # TODO: need an initial value...
        
        self.beta.append(0.)

        # We finally have a temperature where the action is not much smaller than the value we started with (which was
        # close to maxActionThreshold). From here we can use the separation in these temperatures and the action values
        # to predict reasonable temperatures to efficiently sample S/T in the range [minActionThreshold,
        # maxActionThreshold] until the nucleation temperature is found.

        # If we continue to assume linearity, we will overestimate the rate at which minActionThreshold is reached.
        # Therefore, we can take large steps in temperature and dynamically update the step size based on the new
        # prediction of when minActionThreshold will be reached. Further, we can adjust the step size based on how well
        # the prediction matches the observed value. If the prediction is good, we can increase the step size as the
        # action curve must be close to linear in this region. If the prediction is bad, the action must be noticeably
        # non-linear in this region and thus we need a smaller step size for accurate sampling.

        # =================================================
        # Calculate the maximum temperature result.
        # This gives a boundary condition for integration.

        self.nucleationRate[0] = self.calculateBubbleNucleationRate(self.actionSampler.T[0], self.actionSampler.action[0])

        self.numBubblesIntegrand[0] = self.nucleationRate[0] / (self.actionSampler.T[0] * self.hubbleParameter[0] ** 4)
        self.numBubblesCorrectedIntegrand[0] = self.numBubblesIntegrand[0]

        # The temperature for which the action is minimised. This is important for determining the bubble number
        # density. If the minimum action value is encountered after the percolation temperature, then it can be ignored.
        # It is only important if it occurs before percolation.
        TAtActionMin = 0

        # Whether we're handling the first iteration of the overall while loop, handling the interpolation between the
        # first and second action samples. Some things only need to be done in the first iteration as initialisation.
        bFirst = True

        # TODO: add initial bubble radius to these equations and make it a configurable calculation function like
        #  bubbleWallVelocity will be.

        # Outer integrand for the true vacuum volume calculation.
        def outerFunction_trueVacVol(x):
            return self.nucleationRate[x] / (self.actionSampler.subT[x]**4 * self.hubbleParameter[x])

        # Inner integrand for the true vacuum volume calculation.

        def innerFunction_trueVacVol(x):
            return self.bubbleWallVelocity[x] / self.hubbleParameter[x]

        # Outer integrand for the average bubble radius calculation.

        def outerFunction_avgBubRad(x):
            return self.nucleationRate[x] * self.falseVacuumFraction[x] / (self.actionSampler.subT[x]**4
                * self.hubbleParameter[x])

        # Inner integrand for the average bubble radius calculation.
        def innerFunction_avgBubRad(x):
            return self.bubbleWallVelocity[x] / self.hubbleParameter[x]

        # Maps an array index to the temperature corresponding to that index.
        def sampleTransformationFunction(x):
            return self.actionSampler.subT[x]

        # We need three (two) data points before we can initialise the integration helper for the true vacuum volume
        # (average bubble radius). We wait for four data points to be ready (see below for why), then initialise both
        # integration helpers at the same time.

        # The index in the simulation we're up to analysing. Note that we always sample 1 past this index so we can
        # interpolate between T[simIndex] and T[simIndex+1] (the latter is a lower temperature).
        simIndex = 0

        # See the handling of bFirst below for why this is necessary.
        self.actionSampler.subT.append(self.actionSampler.T[0])
        self.actionSampler.subAction.append(self.actionSampler.action[0])

        # The first argument in each of these integration helpers is firstX. The first x value is 0, corresponding to
        # index 0 in the actionSampler.subT list, as per the sampleTransformationFunction.
        self.integrationHelper_trueVacVol = CubedNestedIntegrationHelper(0, outerFunction_trueVacVol,
            innerFunction_trueVacVol, sampleTransformationFunction)
        self.integrationHelper_avgBubRad = LinearNestedNormalisedIntegrationHelper(0, outerFunction_avgBubRad,
            innerFunction_avgBubRad, outerFunction_avgBubRad, sampleTransformationFunction)

        # Keep sampling the action until we have identified the end of the phase transition or that the transition
        # doesn't complete.
        while not self.bCheckPossibleCompletion or self.transitionCouldComplete(maxActionThreshold + toleranceAction,
                self.falseVacuumFraction):
            # If the action begins to increase with decreasing temperature.
            if TAtActionMin == 0 and self.actionSampler.action[simIndex+1] > self.actionSampler.action[simIndex]:
                # TODO: do some form of interpolation. Interp factor will be based on comparing new and old derivs.
                TAtActionMin = self.actionSampler.T[simIndex]
                actionMin = self.actionSampler.action[simIndex]
                self.transition.Tmin = TAtActionMin
                self.transition.actionMin = actionMin

            numSamples = self.actionSampler.getNumSubsamples()
            print('Num samples:', numSamples)
            # List of temperatures in decreasing order.
            T = np.linspace(self.actionSampler.T[simIndex], self.actionSampler.T[simIndex+1], numSamples)
            # We can only use quadratic interpolation if we have at least 3 action samples, which occurs for simIndex > 0.
            if simIndex > 0:
                quadInterp = lagrange(self.actionSampler.T[simIndex-1:], self.actionSampler.action[simIndex-1:])
                action = quadInterp(T)
            else:
                action = np.linspace(self.actionSampler.action[simIndex], self.actionSampler.action[simIndex+1],
                    numSamples)

            #rhof1, rhot1 = hydrodynamics.calculateEnergyDensityAtT(self.fromPhase, self.toPhase, self.potential,
            #    self.actionSampler.T[simIndex])
            #rhof2, rhot2 = hydrodynamics.calculateEnergyDensityAtT(self.fromPhase, self.toPhase, self.potential,
            #    self.actionSampler.T[simIndex+1])
            T1 = self.actionSampler.T[simIndex]
            T2 = self.actionSampler.T[simIndex+1]
            hydroVars1 = self.getHydroVars(T1)
            hydroVars2 = self.getHydroVars(T2)
            hydroVarsInterp = [hydrodynamics.getInterpolatedHydroVars(hydroVars1, hydroVars2, T1, T2, t) for t in T]

            dT = T[0] - T[1]

            # Because we skip the first element of the subsamples in the following loop, make sure to add the very first
            # subsample of the integration to the corresponding lists. This only needs to be done for the first sampling
            # iteration, as the next iteration's first subsample is this iteration's last subsample (i.e. it will
            # already have been added).
            if bFirst:
                # subT and subAction have already been done because they were required for determining numSamples.
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
                self.actionSampler.subAction.append(action[i])
                self.actionSampler.subRhof.append(hydroVarsInterp[i].energyDensityFalse)
                self.actionSampler.subRhot.append(hydroVarsInterp[i].energyDensityTrue)

                # The integration helper needs three data points before it can be initialised. However, we wait for an
                # additional point so that we can immediately integrate and add it to the true vacuum fraction and false
                # vacuum probability arrays as usual below.

                # TODO:
                # Switch to using the initialise method. We can use the naive integration before the integration helpers
                # are initialised, so there should be no need to loop over the recently generated burst of data anymore.
                # TODO:

                """if self.integrationHelper_trueVacVol is None and len(self.actionSampler.subT) == 4:
                    self.integrationHelper_trueVacVol = CubedNestedIntegrationHelper([0, 1, 2],
                        outerFunction_trueVacVol, innerFunction_trueVacVol, sampleTransformationFunction)

                    # Don't add the first element since we have already stored trueVacuumVolumeExtended[0] = 0,
                    # falseVacuumFraction[0] = 1, etc.
                    for j in range(1, len(self.integrationHelper_trueVacVol.data)):
                        self.trueVacuumVolumeExtended.append(4*np.pi/3 * self.integrationHelper_trueVacVol.data[j])
                        self.physicalVolume.append(3 + T[j]*(self.trueVacuumVolumeExtended[-2] -
                            self.trueVacuumVolumeExtended[-1]) / (T[j-1] - T[j]))
                        self.falseVacuumFraction.append(np.exp(-self.trueVacuumVolumeExtended[-1]))
                        self.bubbleNumberDensity.append((T[j]/T[j-1])**3 * self.bubbleNumberDensity[-1]
                            + 0.5*(T[j-1] - T[j])*T[j]**3 * (self.nucleationRate[j-1] * self.falseVacuumFraction[j-1]
                            / (T[j-1]**4 * self.hubbleParameter[j-1]) + self.nucleationRate[j] *
                            self.falseVacuumFraction[j] / (T[j]**4 * self.hubbleParameter[j])))

                    # Do the same thing for the average bubble radius integration helper. This needs to be done after
                    # falseVacuumFraction has been filled with data because outerFunction_avgBubRad uses this data.
                    integrationHelper_avgBubRad = LinearNestedNormalisedIntegrationHelper([0, 1],
                        outerFunction_avgBubRad, innerFunction_avgBubRad, outerFunction_avgBubRad,
                        sampleTransformationFunction)

                    # Since the average bubble radius integration helper requires one less data point for
                    # initialisation, it currently contains one less data point than it should have. Add one more data
                    # point: the previous temperature sample. The current temperature sample will be added later in this
                    # iteration of i.
                    integrationHelper_avgBubRad.integrate(len(self.actionSampler.subT)-2)

                    for j in range(1, len(integrationHelper_avgBubRad.data)):
                        self.meanBubbleRadius.append(integrationHelper_avgBubRad.data[j])
                        self.meanBubbleSeparation.append((self.bubbleNumberDensity[j])**(-1/3))"""

                self.integrateTemperatureStep(T[i], action[i], T[i-1], hydroVarsInterp[i])

                TNew = self.actionSampler.subT[-1]
                #TPrev = self.actionSampler.subT[-2]
                actionNew = self.actionSampler.subAction[-1]
                actionPrev = self.actionSampler.subAction[-2]

                # TODO: not a great derivative, can do better.
                self.beta.append(self.hubbleParameter[-1] * TNew * (actionPrev - actionNew) / dT)

                # Check if we have reached any milestones (e.g. unit nucleation, percolation, etc.).
                self.checkMilestoneTemperatures()

                if self.shouldTerminateTransitionAnalysis():
                    break

            # ==========================================================================================================
            # End subsampling.
            # ==========================================================================================================

            if self.shouldTerminateTransitionAnalysis():
                if self.bDebug:
                    print('Found Tf, stopping sampling')
                break

            if sampleData.T <= self.Tmin:
                if self.bDebug:
                    print('The transition does not complete before reaching Tmin.')
                break

            # Choose the next value of S/T we're aiming to sample.
            #success, message = self.actionSampler.getNextSample(sampleData, self.nucleationRate, self.numBubblesIntegral[-1], self.Tmin)
            success, message = self.actionSampler.getNextSample_new(sampleData)

            simIndex += 1

            if self.timeout_phaseHistoryAnalysis > 0:
                endTime = time.perf_counter()
                if endTime - startTime > self.timeout_phaseHistoryAnalysis:
                    return

            if not success:
                if self.bDebug:
                    print('Terminating transition analysis after failing to get next action sample. Reason:', message)

                if message == 'Freeze out' or message == 'Reached Tmin':
                    break

                if len(precomputedT) > 0:
                    self.transition.analysis.actionCurveFile = precomputedActionCurveFileName
                self.transition.analysis.T = self.actionSampler.T
                self.transition.analysis.action = self.actionSampler.action
                self.transition.analysis.error = f'Failed to get next sample at T={sampleData.T}'
                return

        # ==============================================================================================================
        # End transition analysis.
        # ==============================================================================================================

        # Find the maximum nucleation rate to find TGammaMax. Only do so if there is a minimum in the action.
        if TAtActionMin > 0:
            gammaMaxIndex = np.argmax(self.nucleationRate)
            self.transition.TGammaMax = self.actionSampler.subT[gammaMaxIndex]
            self.transition.actionGammaMax = self.actionSampler.subAction[gammaMaxIndex]
            self.transition.GammaMax = self.nucleationRate[gammaMaxIndex]
            Tlow = self.actionSampler.subT[gammaMaxIndex+1]
            #Tmid = self.actionSampler.subT[gammaMaxIndex]
            Thigh = self.actionSampler.subT[gammaMaxIndex-1]
            Slow = self.actionSampler.subAction[gammaMaxIndex + 1]
            Smid = self.actionSampler.subAction[gammaMaxIndex]
            Shigh = self.actionSampler.subAction[gammaMaxIndex - 1]
            # TODO: should use proper formula for second derivative with non-uniform grid.
            d2SdT2 = (Shigh - 2*Smid + Slow) / (0.5*(Thigh - Tlow))**2
            if self.bDebug:
                print('Calculating betaV, d2SdT2:', d2SdT2)
            if d2SdT2 > 0:
                self.transition.analysis.betaV = self.hubbleParameter[gammaMaxIndex] * self.actionSampler.subT[gammaMaxIndex] * np.sqrt(d2SdT2)

        meanBubbleSeparationAtTp = self.getQuantityAtTemperature(self.meanBubbleSeparation, self.transition.Tp)

        if self.bReportAnalysis:
            print('-------------------------------------------------------------------------------------------')

            print('N(T_0): ', self.numBubblesIntegral[-1])
            print('T_0: ', self.actionSampler.subT[-1])
            print('Tc: ', self.transition.Tc)
            if self.transition.Tn > 0:
                print(f'Unit nucleation at T = {self.transition.Tn}, where S = {self.transition.analysis.actionTn}')
            else:
                print('No unit nucleation.')
            if self.transition.Tp > 0:
                print(f'Percolation at T = {self.transition.Tp}, where S = {self.transition.analysis.actionTp}')
            else:
                print('No percolation. Terminated analysis at T =', self.actionSampler.subT[-1], 'where falseVacuumFraction =', self.falseVacuumFraction[-1])
            if self.transition.Te > 0:
                print(f'trueVacuumVolumeExtended = 1 at T = {self.transition.Te}, where S = {self.transition.analysis.actionTe}')
            if self.transition.Tf > 0:
                print(f'Completion at T = {self.transition.Tf}, where S = {self.transition.analysis.actionTf}')
            else:
                if self.transition.Tp > 0:
                    print('No completion. Terminated analysis at T =', self.actionSampler.subT[-1], 'where falseVacuumFraction =',
                          self.falseVacuumFraction[-1])
                else:
                    print('No completion.')
            if self.transition.TVphysDecr_high > 0:
                if self.transition.TVphysDecr_low > 0:
                    print('Physical volume of the false vacuum decreases between', self.transition.TVphysDecr_low, 'and', self.transition.TVphysDecr_high)
                else:
                    print('Physical volume of the false vacuum decreases below', self.transition.TVphysDecr_high)
            if self.transition.Tn > 0:
                print(f'Reheating from unit nucleation: Treh(Tn): {self.transition.Tn} -> {self.transition.Treh_n}')
            if self.transition.Tp > 0:
                print(f'Reheating from percolation: Treh(Tp): {self.transition.Tp} -> {self.transition.Treh_p}')
            if self.transition.Te > 0:
                print(f'Reheating from trueVacuumVolumeExtended=1: Treh(Te): {self.transition.Te} -> {self.transition.Treh_e}')
            if self.transition.Tf > 0:
                print(f'Reheating from completion: Treh(Tf): {self.transition.Tf} -> {self.transition.Treh_f}')
            if TAtActionMin > 0:
                print('Action minimised at T =', TAtActionMin)
            if self.transition.TGammaMax > 0:
                print('Nucleation rate maximised at T =', self.transition.TGammaMax)

            print(f'Mean bubble separation: {meanBubbleSeparationAtTp:.5e}')

            print('-------------------------------------------------------------------------------------------')

        if self.bDebug:
            print('Total action evaluations:', totalActionEvaluations)

        if self.bPlot:
            self.plotTransitionResults()

        if self.transition.Tp > 0:
            #transitionStrength, _, _ = calculateTransitionStrength(self.potential, self.fromPhase, self.toPhase,
            #    self.transition.Tp)
            # TODO: do this elsewhere, maybe a thermal params file.
            hydroVarsAtTp = hydrodynamics.getHydroVars(self.fromPhase, self.toPhase, self.potential, self.transition.Tp)
            thetaf = (hydroVarsAtTp.energyDensityFalse - hydroVarsAtTp.pressureFalse/hydroVarsAtTp.soundSpeedSqTrue) / 4
            thetat = (hydroVarsAtTp.energyDensityTrue - hydroVarsAtTp.pressureTrue/hydroVarsAtTp.soundSpeedSqTrue) / 4
            transitionStrength = 4*(thetaf - thetat) / (3*hydroVarsAtTp.enthalpyDensityFalse)

            #energyWeightedBubbleRadius, volumeWeightedBubbleRadius = calculateTypicalLengthScale(transition.Tp, indexTp,
            #    indexTn, transition.Tc, meanBubbleSeparationAtTp, bubbleWallVelocity, hubbleParameter, actionSampler.subRhoV, GammaEff, actionSampler.subT,
            #    bPlot=bPlot, bDebug=bDebug)

            self.transition.transitionStrength = transitionStrength
            self.transition.meanBubbleRadius = self.getQuantityAtTemperature(self.meanBubbleRadius, self.transition.Tp)
            self.transition.meanBubbleSeparation = meanBubbleSeparationAtTp
            #transition.energyWeightedBubbleRadius = energyWeightedBubbleRadius
            #transition.volumeWeightedBubbleRadius = volumeWeightedBubbleRadius

            if self.bDebug:
                print('Transition strength:', transitionStrength)

        if self.bReportAnalysis:
            if self.transition.Tp > 0:
                print(f'Mean bubble separation (Tp): {self.transition.meanBubbleSeparation:.5e}')
                print(f'Average bubble radius (Tp):  {self.transition.meanBubbleRadius:5e}')
                print(f'Hubble radius (Tp):          '
                      f'{1 / self.getQuantityAtTemperature(self.hubbleParameter, self.transition.Tp)}')
            if self.transition.Tf > 0:
                print(f'Mean bubble separation (Tf): '
                      f'{(self.getQuantityAtTemperature(self.bubbleNumberDensity, self.transition.Tf))**(-1/3)}')
                print(f'Average bubble radius (Tf):  '
                      f'{self.getQuantityAtTemperature(self.meanBubbleRadius, self.transition.Tf)}')
                print(f'Hubble radius (Tf):          '
                      f'{1 / self.getQuantityAtTemperature(self.hubbleParameter, self.transition.Tf)}')

        if len(precomputedT) > 0:
            self.transition.analysis.actionCurveFile = precomputedActionCurveFileName
        self.transition.analysis.T = self.actionSampler.T
        self.transition.analysis.action = self.actionSampler.action
        self.transition.totalNumBubbles = self.numBubblesIntegral[-1]
        self.transition.totalNumBubblesCorrected = self.numBubblesCorrectedIntegral[-1]

        if self.bComputeSubsampledThermalParams:
            self.transition.TSubampleArray = self.actionSampler.subT
            self.transition.HArray = self.hubbleParameter
            self.transition.betaArray = self.beta
            self.transition.meanBubbleSeparationArray = self.meanBubbleSeparation
            self.transition.meanBubbleRadiusArray = self.meanBubbleRadius
            self.transition.Pf = self.falseVacuumFraction

    # TODO: why pass T and action when the integrator reads them from actionSampler.subT and .subAction?
    def integrateTemperatureStep(self, T, action, TPrev, hydroVars):
        print('------------------------')
        print('integrateTemperatureStep')
        print('------------------------')
        print(f'T = {self.actionSampler.subT[-1]}, S = {self.actionSampler.subAction[-1]}')
        dT: float = TPrev - T
        #hubbleParameter.append(np.sqrt(self.calculateHubbleParameterSq(T[i])))
        #hubbleParameter.append(np.sqrt(calculateHubbleParameterSq_supplied(rhof[i] -
        #   self.groundStateEnergyDensity)))
        self.hubbleParameter.append(np.sqrt(self.calculateHubbleParameterSq_fromHydro(hydroVars)))
        self.bubbleWallVelocity.append(self.getBubbleWallVelocity(hydroVars))

        self.nucleationRate.append(self.calculateBubbleNucleationRate(T, action))

        self.numBubblesIntegrand.append(self.nucleationRate[-1] / (T*self.hubbleParameter[-1]**4))
        self.numBubblesCorrectedIntegrand.append(
            self.nucleationRate[-1] * self.falseVacuumFraction[-1] / (T*self.hubbleParameter[-1]**4))
        self.numBubblesIntegral.append(self.numBubblesIntegral[-1]
            + 0.5*dT*(self.numBubblesIntegrand[-1] + self.numBubblesIntegrand[-2]))
        self.numBubblesCorrectedIntegral.append(self.numBubblesCorrectedIntegral[-1]
            + 0.5*dT*(self.numBubblesCorrectedIntegrand[-1] + self.numBubblesCorrectedIntegrand[-2]))

        #if self.integrationHelper_trueVacVol is not None:
        self.integrationHelper_trueVacVol.integrate(len(self.actionSampler.subT)-1)
        self.trueVacuumVolumeExtended.append(4*np.pi/3 * self.integrationHelper_trueVacVol.data[-1])
        nextToLastT = self.actionSampler.subT[-2] if len(self.actionSampler.subT) > 1 else TPrev + dT
        self.physicalVolume.append(3 + T*(self.trueVacuumVolumeExtended[-2] -
            self.trueVacuumVolumeExtended[-1]) / (nextToLastT - T))
        if self.trueVacuumVolumeExtended[-1] < 100:
            self.falseVacuumFraction.append(np.exp(-self.trueVacuumVolumeExtended[-1]))
        else:
            self.falseVacuumFraction.append(0.)
        self.bubbleNumberDensity.append((T/TPrev)**3 * self.bubbleNumberDensity[-1]
            + 0.5*(TPrev - T)*T**3 * (self.nucleationRate[-2] * self.falseVacuumFraction[-2]
            / (TPrev**4 * self.hubbleParameter[-2]) + self.nucleationRate[-1] *
            self.falseVacuumFraction[-1] / (T ** 4 * self.hubbleParameter[-1])))
        # This needs to be done after the new falseVacuumFraction has been added because outerFunction_avgBubRad uses this data.
        self.integrationHelper_avgBubRad.integrate(len(self.actionSampler.subT)-1)
        self.meanBubbleRadius.append(self.integrationHelper_avgBubRad.data[-1])
        self.meanBubbleSeparation.append((self.bubbleNumberDensity[-1])**(-1/3))

        print(f'Pf new = {self.falseVacuumFraction[-1]}, Pf old = {self.falseVacuumFraction[-2]}, delta = {self.falseVacuumFraction[-2] - self.falseVacuumFraction[-1]}')

    def undoIntegration(self, num_iters: int):
        for _ in range(num_iters):
            self.hubbleParameter.pop()
            self.bubbleWallVelocity.pop()
            self.nucleationRate.pop()
            self.numBubblesIntegrand.pop()
            self.numBubblesCorrectedIntegrand.pop()
            self.numBubblesIntegral.pop()
            self.numBubblesCorrectedIntegral.pop()

            #self.integrationHelper_trueVacVol.undo()
            self.trueVacuumVolumeExtended.pop()
            self.physicalVolume.pop()
            self.falseVacuumFraction.pop()
            self.bubbleNumberDensity.pop()
            #self.integrationHelper_avgBubRad.undo()
            self.meanBubbleRadius.pop()
            self.meanBubbleSeparation.pop()

        self.integrationHelper_trueVacVol.restore()
        self.integrationHelper_avgBubRad.restore()

    def plotTransitionResults(self):
        plt.rcParams.update({"text.usetex": True})

        GammaEff = np.array([self.falseVacuumFraction[i] * self.nucleationRate[i] for i in
            range(len(self.nucleationRate))])

        plt.plot(self.actionSampler.subT, self.bubbleWallVelocity)
        plt.xlabel('T')
        plt.ylabel('v_w')
        plt.ylim(0, 1)
        plt.show()

        if self.transition.Tf > 0:
            indexTf: int = 0

            for i in range(len(self.actionSampler.subT)):
                if self.actionSampler.subT[i] < self.transition.Tf:
                    indexTf = i
                    break

            maxIndex = len(self.actionSampler.subT)
            maxIndex = min(len(self.actionSampler.subT) - 1, maxIndex - (maxIndex - indexTf) // 2)
            physicalVolumeRelative = [
                100 * (self.transition.Tf / self.actionSampler.subT[i]) ** 3 * self.falseVacuumFraction[i]
                for i in range(maxIndex + 1)]

            ylim = np.array(physicalVolumeRelative[:min(indexTf + 1, maxIndex)]).max(initial=1.) * 1.2

            textXOffset = 0.01 * (self.actionSampler.subT[0] - self.actionSampler.subT[maxIndex])
            textY = 0.1

            plt.figure(figsize=(14, 11))
            plt.plot(self.actionSampler.subT[:maxIndex + 1], physicalVolumeRelative, zorder=3, lw=3.5)
            # if TVphysDecr_high > 0: plt.axvline(TVphysDecr_high, c='r', ls='--', lw=2)
            # if TVphysDecr_low > 0: plt.axvline(TVphysDecr_low, c='r', ls='--', lw=2)
            if self.transition.TVphysDecr_high > 0 and self.transition.TVphysDecr_low > 0:
                plt.axvspan(self.transition.TVphysDecr_low, self.transition.TVphysDecr_high, alpha=0.3, color='r',
                            zorder=-1)
            if self.transition.Tp > 0:
                plt.axvline(self.transition.Tp, c='g', ls='--', lw=2)
                plt.text(self.transition.Tp + textXOffset, textY, '$T_p$', fontsize=44, horizontalalignment='left')
            if self.transition.Te > 0:
                plt.axvline(self.transition.Te, c='b', ls='--', lw=2)
                plt.text(self.transition.Te + textXOffset, textY, '$T_e$', fontsize=44, horizontalalignment='left')
            if self.transition.Tf > 0:
                plt.axvline(self.transition.Tf, c='k', ls='--', lw=2)
                plt.text(self.transition.Tf - textXOffset, textY, '$T_f$', fontsize=44, horizontalalignment='right')
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
            # saveFolder = 'C:/Work/Monash/PhD/Documents/Subtleties of supercooled cosmological first-order phase transitions/images/'
            # plt.savefig(saveFolder + 'Relative physical volume.png', bbox_inches='tight', pad_inches=0.1)

        ylim = np.array(self.physicalVolume).min(initial=0.)
        ylim *= 1.2 if ylim < 0 else 0.8
        if -0.5 < ylim < 0:
            ylim = -0.5

        textXOffset = 0.01 * (self.actionSampler.subT[0] - 0)
        textY = ylim + 0.07 * (3.5 - ylim)

        plt.figure(figsize=(14, 11))
        plt.plot(self.actionSampler.subT, self.physicalVolume, zorder=3, lw=3.5)
        # if TVphysDecr_high > 0: plt.axvline(TVphysDecr_high, c='r', ls='--', lw=2)
        # if TVphysDecr_low > 0: plt.axvline(TVphysDecr_low, c='r', ls='--', lw=2)
        if self.transition.TVphysDecr_high > 0 and self.transition.TVphysDecr_low > 0:
            plt.axvspan(self.transition.TVphysDecr_low, self.transition.TVphysDecr_high, alpha=0.3, color='r',
                        zorder=-1)
        if self.transition.Tp > 0:
            plt.axvline(self.transition.Tp, c='g', ls='--', lw=2)
            plt.text(self.transition.Tp + textXOffset, textY, '$T_p$', fontsize=44, horizontalalignment='left')
        if self.transition.Te > 0:
            plt.axvline(self.transition.Te, c='b', ls='--', lw=2)
            plt.text(self.transition.Te + textXOffset, textY, '$T_e$', fontsize=44, horizontalalignment='left')
        if self.transition.Tf > 0:
            plt.axvline(self.transition.Tf, c='k', ls='--', lw=2)
            plt.text(self.transition.Tf - textXOffset, textY, '$T_f$', fontsize=44, horizontalalignment='right')
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
        # saveFolder = 'C:/Work/Monash/PhD/Documents/Subtleties of supercooled cosmological first-order phase transitions/images/'
        # plt.savefig(saveFolder + 'Decreasing physical volume.png', bbox_inches='tight', pad_inches=0.1)

        Tn = self.transition.Tn
        Tp = self.transition.Tp
        Te = self.transition.Te
        Tf = self.transition.Tf
        actionTn = self.transition.analysis.actionTn
        actionTp = self.transition.analysis.actionTp
        actionTe = self.transition.analysis.actionTe
        actionTf = self.transition.analysis.actionTf
        # plt.rcParams.update({"text.usetex": True})
        # plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']  # for \text command

        plt.figure(figsize=(12, 8))
        plt.plot(self.actionSampler.subT, self.nucleationRate)
        plt.plot(self.actionSampler.subT, GammaEff)
        # plt.plot(actionSampler.subT, approxGamma)
        # plt.plot(actionSampler.subT, taylorExpGamma)
        plt.xlabel('$T$', fontsize=24)
        plt.ylabel('$\\mathrm{Nucleation rate (GeV^4)}$', fontsize=24)
        if self.transition.TGammaMax > 0: plt.axvline(self.transition.TGammaMax, c='g', ls=':')
        if self.transition.Tmin > 0: plt.axvline(self.transition.Tmin, c='r', ls=':')
        if Tp > 0: plt.axvline(Tp, c='g', ls=':')
        if Te > 0: plt.axvline(Te, c='b', ls=':')
        if Tf > 0: plt.axvline(Tf, c='k', ls=':')
        plt.legend(['$\\mathrm{standard}$', '$\\mathrm{effective}$'], fontsize=24)
        plt.tick_params(size=5, labelsize=16)
        plt.margins(0, 0)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.plot(self.actionSampler.subT, self.bubbleNumberDensity, linewidth=2.5)
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
            if self.meanBubbleSeparation[i] <= self.meanBubbleSeparation[-1]:
                highTempIndex = i
                break

        # If the mean bubble separation is always larger than it is at the lowest sampled temperature, plot the
        # entire range of sampled temperatures.
        if highTempIndex == len(self.actionSampler.subT) - 1:
            highTempIndex = 0

        plt.figure(figsize=(12, 8))
        plt.plot(self.actionSampler.subT, self.meanBubbleRadius, linewidth=2.5)
        plt.plot(self.actionSampler.subT, self.meanBubbleSeparation, linewidth=2.5)
        plt.xlabel('$T \, \\mathrm{[GeV]}$', fontsize=24)
        # plt.ylabel('$\\overline{R}_B(T)$', fontsize=24)
        plt.legend(['$\\overline{R}_B(T)$', '$R_*(T)$'], fontsize=24)
        if Tn > 0: plt.axvline(Tn, c='r', ls=':')
        if Tp > 0: plt.axvline(Tp, c='g', ls=':')
        if Te > 0: plt.axvline(Te, c='b', ls=':')
        if Tf > 0: plt.axvline(Tf, c='k', ls=':')
        plt.xlim(self.actionSampler.subT[-1], self.actionSampler.subT[highTempIndex])
        plt.ylim(0, 1.2 * max(self.meanBubbleSeparation[-1], self.meanBubbleRadius[-1]))
        plt.tick_params(size=5, labelsize=16)
        plt.margins(0, 0)
        plt.tight_layout()
        plt.show()

        highTempIndex = 0
        lowTempIndex = len(self.actionSampler.action)

        minAction = min(self.actionSampler.action)
        maxAction = self.actionSampler.maxActionThreshold

        # Search for the lowest temperature for which the action is not significantly larger than the maximum
        # significant action.
        for i in range(len(self.actionSampler.action)):
            if self.actionSampler.action[i] <= maxAction:
                highTempIndex = i
                break

        # Search for the lowest temperature for which the action is not significantly larger than the maximum
        # significant action.
        for i in range(len(self.actionSampler.action) - 1, -1, -1):
            if self.actionSampler.action[i] <= maxAction:
                lowTempIndex = i
                break

        plt.figure(figsize=(12, 8))
        plt.plot(self.actionSampler.subT, self.actionSampler.subAction, linewidth=2.5)
        # plt.plot(actionSampler.subT, approxAction, linewidth=2.5)
        plt.scatter(self.actionSampler.T, self.actionSampler.action)
        if Tn > -1:
            plt.axvline(Tn, c='r', ls=':')
            plt.axhline(actionTn, c='r', ls=':')
        if Tp > -1:
            plt.axvline(Tp, c='g', ls=':')
            plt.axhline(actionTp, c='g', ls=':')
        if Te > -1:
            plt.axvline(Te, c='b', ls=':')
            plt.axhline(actionTe, c='b', ls=':')
        if Tf > -1:
            plt.axvline(Tf, c='k', ls=':')
            plt.axhline(actionTf, c='k', ls=':')
        plt.minorticks_on()
        plt.grid(visible=True, which='major', color='k', linestyle='--')
        plt.grid(visible=True, which='minor', color='gray', linestyle=':')
        plt.xlabel('$T \, \\mathrm{[GeV]}$', fontsize=24)
        plt.ylabel('$S(T)$', fontsize=24)
        # plt.legend(['precise', 'approx'])
        plt.xlim(self.actionSampler.T[lowTempIndex], self.actionSampler.T[highTempIndex])
        plt.ylim(minAction - 0.05 * (maxAction - minAction), maxAction)
        plt.tick_params(size=5, labelsize=16)
        plt.margins(0, 0)
        plt.tight_layout()
        plt.show()

        # Number of bubbles plotted over entire sampled temperature range, using log scale for number of bubbles.
        plt.figure(figsize=(12, 8))
        plt.plot(self.actionSampler.subT, self.numBubblesCorrectedIntegral, linewidth=2.5)
        plt.plot(self.actionSampler.subT, self.numBubblesIntegral, linewidth=2.5)
        # plt.plot(actionSampler.subT, numBubblesApprox, linewidth=2.5)
        if Tn > 0: plt.axvline(Tn, c='r', ls=':')
        if Tp > 0: plt.axvline(Tp, c='g', ls=':')
        if Te > 0: plt.axvline(Te, c='b', ls=':')
        if Tf > 0: plt.axvline(Tf, c='k', ls=':')
        plt.xlabel('$T \, \\mathrm{[GeV]}$', fontsize=24)
        # plt.ylabel('$N(T)$', fontsize=24)
        plt.yscale('log')
        # plt.legend(['precise', 'approx'])
        plt.legend(['$N(T)$', '$N^{\\mathrm{ext}}(T)$'], fontsize=24)
        plt.tick_params(size=5, labelsize=16)
        plt.margins(0, 0)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.plot(self.actionSampler.subT, self.falseVacuumFraction, linewidth=2.5)
        if Tn > 0: plt.axvline(Tn, c='r', ls=':')
        if Tp > 0: plt.axvline(Tp, c='g', ls=':')
        if Te > 0: plt.axvline(Te, c='b', ls=':')
        if Tf > 0: plt.axvline(Tf, c='k', ls=':')
        plt.xlabel('$T \, \\mathrm{[GeV]}$', fontsize=40)
        plt.ylabel('$P_f(T)$', fontsize=40)
        plt.tick_params(size=8, labelsize=28)
        plt.margins(0, 0)
        plt.tight_layout()
        plt.show()
        # plt.savefig("output/plots/Pf_vs_T_BP2.pdf")
        # plt.savefig('E:/Monash/PhD/Milestones/Confirmation Review/images/xSM P(T) vs T.png', bbox_inches='tight',
        #    pad_inches=0.05)

    def getQuantityAtTemperature(self, quantity: List[float], temperature: float) -> float:
        index: int = 0

        # The requested temperature is above the maximum sampled temperature.
        if temperature > self.actionSampler.subT[0]:
            if self.bDebug:
                print(f'Attempted to get quantity at temperature{temperature}, above the maximum sampled temperature, '
                      f'{self.actionSampler.subT[0]}. Returning the quantity at the maximum sampled temperature.')
            return quantity[0]

        # The requested temperature is below the minimum sampled temperature.
        if temperature < self.actionSampler.subT[-1]:
            if self.bDebug:
                print(f'Attempted to get quantity at temperature{temperature}, below the minimum sampled temperature, '
                      f'{self.actionSampler.subT[-1]}. Returning the quantity at the minimum sampled temperature.')
            return quantity[-1]

        # Step through the temperature samples in decreasing order.
        for i in range(len(self.actionSampler.subT)):
            # We can return the exact sampled quantity if the requested temperature matches a sample point.
            if self.actionSampler.subT[i] == temperature:
                return quantity[i]
            # Otherwise, check if the sample temperatures are now too low.
            elif self.actionSampler.subT[i] < temperature:
                index = i
                break

        # Use linear interpolation to estimate the quantity at the requested temperature.
        interpFactor = (temperature - self.actionSampler.subT[index]) / (self.actionSampler.subT[index] -
            self.actionSampler.subT[index-1])
        return quantity[index] + interpFactor*(quantity[index-1] - quantity[index])

    # TODO: a lot of this needs updating.
    def primeTransitionAnalysis(self, startTime: float) -> (Optional[ActionSample], list[ActionSample]):
        TcData: ActionSample = ActionSample(self.transition.Tc)
        TminData: ActionSample = ActionSample(self.Tmin)

        if self.bDebug:
            # \u00B1 is the plus/minus symbol.
            print('Bisecting to find S/T =', self.actionSampler.maxActionThreshold, u'\u00B1',
                  self.actionSampler.actionTargetTolerance)

        # Use bisection to find the temperature at which S/T ~ maxActionThreshold.
        actionCurveShapeAnalysisData: ActionCurveShapeAnalysisData =\
            self.findNucleationTemperatureWindow_refined(startTime=startTime)

        if self.timeout_phaseHistoryAnalysis > 0:
            endTime: float = time.perf_counter()
            if endTime - startTime > self.timeout_phaseHistoryAnalysis:
                return None, []

        # TODO: Maybe use the names from here anyway? The current set of names is a little confusing!
        data: ActionSample = actionCurveShapeAnalysisData.desiredData
        bisectMinData: ActionSample = actionCurveShapeAnalysisData.nextBelowDesiredData
        lowerActionData: List[ActionSample] = actionCurveShapeAnalysisData.storedLowerActionData
        allSamples: List[ActionSample] = actionCurveShapeAnalysisData.actionSamples
        bBarrierAtTmin: bool = actionCurveShapeAnalysisData.bBarrierAtTmin

        # If we didn't find any action values near the nucleation threshold, we are done.
        if data is None or not data.bValid:
            if len(allSamples) == 0:
                if self.bReportAnalysis:
                    print('No transition')
                    print('No action samples')
                return None, []

            #allSamples = np.array(allSamples)
            #minActionIndex = np.argmin(allSamples[:, 1])

            # Find the minimum action so we can record it.
            minActionIndex = 0
            minAction = allSamples[0].action

            for i in range(1, len(allSamples)):
                if allSamples[i].action < minAction:
                    minAction = allSamples[i].action
                    minActionIndex = i

            self.transition.analysis.lowestAction = minAction

            if self.bReportAnalysis:
                print('No transition')
                print('Lowest sampled S/T =', minAction, 'at T =', allSamples[minActionIndex].T)

            # This is a little confusing at first, but what this does is:
            # - allSamples.argsort(axis=0): sort the indices of both columns (T and S/T) based on the values stored in
            #   the corresponding positions.
            # - [:,0]: grab the first column (the T column).
            # - Use the sorted indices based on T to grab the elements of allSamples in the correct order.
            # This returns a copy of allSamples with the same (T, S/T) tuples, but sorted by the T values.
            #allSamples = allSamples[allSamples.argsort(axis=0)[:, 0]]
            # TODO: [2024] changed to use the key parameter in regular sort. Need to make sure this works as intended.
            allSamples.sort(key=lambda x: x.T, reverse=True)

            T = [sample.T for sample in allSamples]
            action = [sample.action for sample in allSamples]

            if self.bDebug:
                print('T:', T)
                print('action:', action)

            if self.bPlot:
                plt.plot(T, action)
                plt.xlabel('$T$')
                plt.ylabel('$S(T)$')
                plt.yscale('log')
                # plt.ylim(bottom=2)
                plt.show()

            self.transition.analysis.T = T
            self.transition.analysis.action = action
            return None, []
        else:
            if self.bDebug:
                print(f'Data: (T: {data.T}, S: {data.action})')
                print('len(lowerActionData):', len(lowerActionData))

        intermediateData: ActionSample = ActionSample.copyData(data)

        self.actionSampler.lowerActionData = lowerActionData

        if self.bDebug:
            print('Attempting to find next reasonable (T, S/T) sample below maxActionThreshold...')

        if data.action > self.actionSampler.minActionThreshold + 0.8*(self.actionSampler.maxActionThreshold -
                                                                      self.actionSampler.minActionThreshold):
            if self.bDebug:
                print('Presumably not a subcritical transition curve, with current action near maxActionThreshold.')
            subcritical = False
            # targetAction is the first action value we would like to sample. Skipping this might lead to numerical
            # errors in the integration, and sampling at higher S/T values is numerically insignificant.
            # targetAction = minActionThreshold + 0.98*(min(maxActionThreshold, data.action) - minActionThreshold)
            targetAction = data.action * self.actionSampler.stepSizeMax

            # Check if the bisection window can inform which temperature we should sample to reach the target S/T.
            if abs(bisectMinData.action - targetAction) < 0.3 * (
                    self.actionSampler.maxActionThreshold - self.actionSampler.minActionThreshold):
                interpFactor = (targetAction - bisectMinData.action) / (data.action - bisectMinData.action)
                intermediateData.T = bisectMinData.T + interpFactor * (data.T - bisectMinData.T)
            # Otherwise, check if the low S/T data can inform which temperature we should sample to reach the target S/T.
            elif len(lowerActionData) > 0 and abs(lowerActionData[-1].action - targetAction) \
                    < 0.5 * (self.actionSampler.maxActionThreshold - self.actionSampler.minActionThreshold):
                interpFactor = (targetAction - lowerActionData[-1].action) / (data.action - lowerActionData[-1].action)
                intermediateData.T = lowerActionData[-1].T + interpFactor * (data.T - lowerActionData[-1].T)
            # Otherwise, all that's left to do is guess.
            else:
                # Try sampling S/T at a temperature just below where S/T = maxActionThreshold, and determine where to sample
                # next based on the result.
                intermediateData.T = self.Tmin + 0.99*(data.T - self.Tmin)

            self.actionSampler.evaluateAction(intermediateData)

            if self.timeout_phaseHistoryAnalysis > 0:
                endTime = time.perf_counter()
                if endTime - startTime > self.timeout_phaseHistoryAnalysis:
                    return None, []

            if not intermediateData.bValid:
                self.transition.analysis.error = f'Failed to evaluate action at trial temperature T={intermediateData.T}'
                return None, []

            # If we happen to sample too far away from 'data' (which can happen if S/T is very steep near maxActionThreshold),
            # then we should correct our next sample to be closer to maxActionThreshold. In case of a noisy action, make sure
            # to limit the number of samples and simply choose the result with the closest S/T to maxActionThreshold.
            maxCorrectionSamples = 5
            correctionSamplesTaken = 0
            closestPoint = ActionSample.copyData(intermediateData)

            # While our sample's S/T is too far from the target value, step closer to 'data' and try again.
            while correctionSamplesTaken < maxCorrectionSamples \
                    and abs(1 - abs(intermediateData.action - data.action) / data.action) < self.actionSampler.stepSizeMax:
                if self.bDebug:
                    print('Sample too far from target S/T value at T =', intermediateData.T, 'with S/T =',
                          intermediateData.action)
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
                self.actionSampler.storeLowerActionData(ActionSample.copyData(intermediateData))

                # If this is the closest point, store this in case the next sample is worse (for a noisy action).
                if abs(intermediateData.action - data.action) < abs(closestPoint.action - data.action):
                    intermediateData.transferData(closestPoint)

            # If we corrected intermediate data, make sure to update it to the closest point (to maxActionThreshold) sampled.
            if correctionSamplesTaken > 0:
                closestPoint.transferData(intermediateData)

            # Given that we took a small step in temperature and have a relatively large S/T, an increase in S/T means there
            # is insufficient time for nucleation to occur. It is improbable that S/T would drop to a small enough value
            # within this temperature range to yield nucleation.
            if intermediateData.action >= data.action:
                if self.bDebug:
                    print('S/T increases before nucleation can occur.')
                return None, self.actionSampler.lowerActionData

            self.actionSampler.T.extend([data.T, intermediateData.T])
            self.actionSampler.action.extend([data.action, intermediateData.action])
        else:
            if self.bDebug:
                print(
                    'Presumably a subcritical transition curve, with current S/T significantly below maxActionThreshold.')
            subcritical = True

            # Take a very small step, with the size decreasing as S/T decreases.
            interpFactor = 0.99 + 0.009*(1 - data.action / self.actionSampler.minActionThreshold)
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
            if not intermediateData.bValid or intermediateData.action == 0 and TcData.action == 0:
                self.transition.analysis.error = f'Failed to evaluate action at trial temperature T={intermediateData.T}'
                # print('This was for a subcritical transition with initial S/T:', data.action, 'at T:', data.T)
                return None, []

            # If we couldn't sample all the way to Tmax, predict what the action would be at Tmax and store that as a
            # previous sample to be used in the following integration. intermediateData will be used as the next sample
            # point.
            if data.T < self.Tmax:
                maxAction = data.action + (data.action - intermediateData.action) * (self.Tmax - data.T) / (
                            data.T - intermediateData.T)

                self.actionSampler.T.extend([self.Tmax, data.T])
                self.actionSampler.action.extend([maxAction, data.action])

                # We have already sampled this data point and should use it as the next point in the integration. Storing it
                # in lowerActionData automatically results in this desired behaviour. No copy is required as we don't alter
                # intermediateData from this point on.
                self.actionSampler.storeLowerActionData(intermediateData)
            # If we sampled all the way to Tmax, use the sample there and intermediateData as the samples for the following
            # integration.
            else:
                self.actionSampler.T.extend([data.T, intermediateData.T])
                self.actionSampler.action.extend([data.action, intermediateData.action])

        if self.bDebug:
            print('Found next sample: T =', intermediateData.T, 'and S =', intermediateData.action)

        # Now take the same step in temperature and evaluate the action again.
        sampleData = ActionSample.copyData(intermediateData)
        sampleData.T = 2 * intermediateData.T - data.T

        if not subcritical and len(self.actionSampler.lowerActionData) > 0 and self.actionSampler.lowerActionData[-1].T\
                >= sampleData.T:
            if self.actionSampler.lowerActionData[-1].T < self.actionSampler.T[-1]:
                self.actionSampler.lowerActionData[-1].transferData(sampleData)
            else:
                self.actionSampler.evaluateAction(sampleData)

                if self.timeout_phaseHistoryAnalysis > 0:
                    endTime = time.perf_counter()
                    if endTime - startTime > self.timeout_phaseHistoryAnalysis:
                        return None, []

            self.actionSampler.lowerActionData.pop()
        else:
            sampleData.action = -1
            sampleData.action = -1

            self.actionSampler.evaluateAction(sampleData)

            if self.timeout_phaseHistoryAnalysis > 0:
                endTime = time.perf_counter()
                if endTime - startTime > self.timeout_phaseHistoryAnalysis:
                    return None, []

            if not sampleData.bValid:
                print('Failed to evaluate action at trial temperature T=', sampleData.T)

        self.actionSampler.calculateStepSize(low=sampleData, mid=intermediateData, high=data)

        # We have already sampled this data point and should use it as the next point in the integration. Storing it in
        # lowerActionData automatically results in this desired behaviour. For a near-instantaneous subcritical transition,
        # this will actually be the *second* next point in the integration, as the handling of intermediateData is also
        # postponed, and should be done before sampleData.
        self.actionSampler.storeLowerActionData(ActionSample.copyData(sampleData))

        return sampleData, allSamples

    # TODO: This currently is based on a crude estimate of the nucleation temperature. The nucleation temperature
    #  corresponds to 1 bubble per Hubble volume. Assume a constant action (i.e. independent of temperature). What
    #  (constant) action value leads to roughly 1 bubble per Hubble volume? This informs the action scale of the
    #  phase transition.
    #  Because this is such a crude approximation, and because the nucleation temperature is not a good starting point
    #  for transition analysis in general, the actual maximum action scale is taken to be the returned value plus some
    #  large correction to ensure we don't miss some important details about the start of the phase transition.
    #  I think we should change this to be some (over-)estimate of the true vacuum fraction and we can e.g. search for
    #  P_t(S) ~ 0.1%.
    def estimateMaximumSignificantAction(self, tolerance: float = 2.0):
        # TODO: these need to be configurable or start with larger variation.
        actionMin: float = 50
        actionMax: float = 200
        action: float = 160

        while actionMax - actionMin > tolerance:
            action = 0.5*(actionMin + actionMax)

            nucRate: float = self.calculateInstantaneousNucleationRate(self.Tmax, action)
            numBubbles: float = nucRate*(self.Tmax - self.Tmin)

            if 0.1 < numBubbles < 10:
                return action
            elif numBubbles < 1:
                actionMax = action
            else:
                actionMin = action

        return action

    def findNucleationTemperatureWindow_refined(self, startTime: float = -1.0):
        actionCurveShapeAnalysisData: ActionCurveShapeAnalysisData = ActionCurveShapeAnalysisData()
        Tstep: float = 0.01*(self.Tmax - self.Tmin)
        actionSamples: List[ActionSample] = []
        lowerActionData = []

        TLow = self.Tmin
        THigh = self.Tmax

        # ==============================================================================================================
        # Begin locally-defined convenience functions.
        # ==============================================================================================================

        # Wrapper function that evaluates the action and stores it in data. Returns whether the action was successfully
        # evaluated.
        def evaluateAction_internal() -> bool:
            try:
                self.actionSampler.evaluateAction(data)
            except ThinWallError:
                if not self.bAllowErrorsForTn:
                    return False
            except Exception:
                import traceback
                traceback.print_exc()
                return False

            #actionSamples.append((data.T, data.action, data.bValid))
            actionSamples.append(ActionSample.copyData(data))

            if data.bValid and self.actionSampler.minActionThreshold < data.action < self.actionSampler.maxActionThreshold:
                lowerActionData.append(ActionSample.copyData(data))

            return True

        # Returns a label that indicates whether the action store in data is above or below the target value.
        def getSignLabel() -> str:
            if not data.bValid:
                return 'unknown'
            elif abs(data.action - self.actionSampler.maxActionThreshold) <= self.actionSampler.actionTargetTolerance:
                return 'equal'
            elif data.action > self.actionSampler.maxActionThreshold:
                return 'above'
            else:
                return 'below'

        def saveCurveShapeAnalysisData(bSuccess: bool = False):
            dataToUse = data
            lowerTSampleToUse = lowerTSample

            # If the transition is said to fail (e.g. due to failure to evaluate the bounce in the thin-walled limit),
            # check if we have enough data to suggest the transition should occur.
            if not bSuccess:
                # Find the highest temperature for which the action is below the nucleation threshold.
                maxTempBelowIndex = -1
                prevMaxTempBelowIndex = -1
                for i in range(len(actionSamples)):
                    if actionSamples[i].bValid and actionSamples[i].action < self.actionSampler.maxActionThreshold:
                        if maxTempBelowIndex < 0 or actionSamples[i].T > actionSamples[maxTempBelowIndex].T:
                            prevMaxTempBelowIndex = maxTempBelowIndex
                            maxTempBelowIndex = i

                # If such a sample exists, we can use this as the desired data point.
                if maxTempBelowIndex > -1:
                    bSuccess = True
                    temperature: float = actionSamples[maxTempBelowIndex].T
                    action = actionSamples[maxTempBelowIndex].action
                    dataToUse = ActionSample(temperature, action*temperature)
                    dataToUse.bValid = True

                    # Attempt to also store the sample point directly before this in temperature.
                    if prevMaxTempBelowIndex > -1:
                        temperature = actionSamples[prevMaxTempBelowIndex].T
                        action = actionSamples[prevMaxTempBelowIndex].action
                        lowerTSampleToUse = ActionSample(temperature, action*temperature)
                        lowerTSampleToUse.bValid = True
                    else:
                        lowerTSampleToUse = None

            if bSuccess:
                actionCurveShapeAnalysisData.copyDesiredData(dataToUse)
                actionCurveShapeAnalysisData.copyNextBelowDesiredData(lowerTSampleToUse)
                # We probably just stored data in lowerActionData, so remove it since we're using it as the target value.
                if self.actionSampler.minActionThreshold < dataToUse.action < self.actionSampler.maxActionThreshold:
                    lowerActionData.pop()
            actionCurveShapeAnalysisData.copyActionSamples(actionSamples)
            actionCurveShapeAnalysisData.copyStoredLowerActionData(lowerActionData)

        # ==============================================================================================================
        # End locally-defined convenience functions.
        # ==============================================================================================================

        # Evaluate the action at Tmin. (Actually just above so we avoid issues where the phase might disappear.)
        data = ActionSample(self.Tmin+Tstep)
        lowerTSample = ActionSample(self.Tmin+Tstep)
        bSuccess = evaluateAction_internal()

        if not bSuccess:
            saveCurveShapeAnalysisData(False)
            return actionCurveShapeAnalysisData

        labelLow = getSignLabel()

        # If the action calculation failed at the low temperature, step along the curve slightly and try again.
        # TODO: [2024] maybe leave this as configurable alternative behaviour? It would be used as an alternative to the
        #  uncommented code just below.
        """if labelLow == 'unknown':
            data.T += 5*Tstep
            evaluateAction()
            labelLow = getSignLabel()
    
            # If we still can't calculate the action, give up.
            if labelLow == 'unknown':
                saveCurveShapeAnalysisData()
                return actionCurveShapeAnalysisData
    
            TLow = data.T"""

        # Assume that if the action calculation fails, it's probably because the barrier becomes too small, hence the
        # action would be very low.
        if labelLow == 'unknown':
            if self.bAllowErrorsForTn:
                labelLow = 'below'
            # else we would have returned already.

        # If the action is high at Tmin.
        if labelLow == 'above' or labelLow == 'equal':
            # Check at a slightly higher temperature to determine whether the action is increasing or decreasing.

            # Store the result at Tmin to compare to.
            minT = data.T
            minAction = data.action
            minLabel = labelLow

            data.transferData(lowerTSample)
            data.T += Tstep
            bSuccess = evaluateAction_internal()

            if not bSuccess:
                saveCurveShapeAnalysisData(False)
                return actionCurveShapeAnalysisData

            if data.bValid:
                labelLow = getSignLabel()
                TLow = data.T

                # If the action is increasing (going from Tmin to Tmin+deltaT), then we will not find a lower action.
                if data.action > minAction:
                    # If the action at the minimum temperature was equal to the threshold within tolerance, we have shown
                    # that it was the minimum action overall and can return it as the only solution.
                    if minLabel == 'equal':
                        data.T = minT
                        data.action = minAction
                        actionCurveShapeAnalysisData.copyDesiredData(data)
                    else:
                        actionCurveShapeAnalysisData.confidentNoNucleation = True

                    saveCurveShapeAnalysisData()
                    return actionCurveShapeAnalysisData

        # Evaluate the action at Tmax if this transition is not evaluated at the critical temperature. Otherwise we already
        # know the action is divergent at the critical temperature.
        if self.Tmax < self.transition.Tc or self.transition.subcritical:
            if self.Tmax < self.transition.Tc:
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
                data.transferData(lowerTSample)
                data.T = 0.5*(data.T + self.Tmax)
                Tstep = 0.1*(self.Tmax - data.T)
                bSuccess = evaluateAction_internal()

                if not bSuccess:
                    saveCurveShapeAnalysisData(False)
                    return actionCurveShapeAnalysisData

                # TODO: cleanup if this still works [2023]
                #calculateActionSimple(potential, data, fromPhase, toPhase, Tstep=-Tstep, bDebug=settings.bDebug)
                #actionSamples.append((data.T, data.action, data.bValid))
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
            data.transferData(lowerTSample)
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

            # The temperature stored in data might be slightly different form TMid if there was a failed action
            # calculation and a later attempt was successful.
            midT = data.T
            midAction = data.action
            data.transferData(lowerTSample)
            data.T += Tstep
            bSuccess = evaluateAction_internal()

            if not bSuccess:
                saveCurveShapeAnalysisData(False)
                return actionCurveShapeAnalysisData

            if not data.bValid:
                if self.bAllowErrorsForTn:
                    TLow = TMid
                # else we would have returned already.
            elif data.action > midAction:
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
            # target value is reached (based on the possible shape of the action curve), if this predicted temperature
            # lies outside of the current temperature window, we can conservatively claim that the target value cannot
            # be found.
            dTdS = (data.T - midT) / (data.action - midAction)

            if data.action > midAction:
                TPredicted = midT - (midAction - self.actionSampler.maxActionThreshold) * dTdS
                extrapolationFailed = TPredicted < TLow
            else:
                TPredicted = data.T - (data.action - self.actionSampler.maxActionThreshold) * dTdS
                extrapolationFailed = TPredicted > THigh

            if extrapolationFailed:
                if self.bDebug:
                    print('Action will not go below', self.actionSampler.maxActionThreshold,
                          'based on linear extrapolation.')

                saveCurveShapeAnalysisData()
                actionCurveShapeAnalysisData.confidentNoNucleation = True
                return actionCurveShapeAnalysisData

            TMid = 0.5*(TLow + THigh)

        # Now we should have labelLow='below' and labelHigh='above'. That is, we bracket the desired solution. We can
        # simply bisect from here.
        while THigh - TLow > (self.Tmax - self.Tmin)*1e-5:
            data.transferData(lowerTSample)
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

    def calculateBubbleNucleationRate(self, T: Union[float, np.ndarray], action: Union[float, np.ndarray])\
            -> Union[float, np.ndarray]:
        #print('calculateBubbleNucleationRate:', T, action, T**4 * (action/(2*np.pi))**(3/2) * np.exp(-action))
        # Prevent underflow.
        if np.isscalar(T):
            if action < 500:
                return T**4 * (action/(2*np.pi))**(3/2) * np.exp(-action)
            else:
                return 0.
        else:
            nucleationRate = np.zeros(shape=T.shape)
            mask = action < 500
            nucleationRate[mask] = T[mask]**4 * (action[mask]/(2*np.pi))**(3/2) * np.exp(-action[mask])
            return nucleationRate

    # TODO: [2023] need to allow this to be used for list inputs again. [2024] That would require making
    #  hydrodynamics.calculateEnergyDensityAtT_singlePhase accept list inputs too. Not necessary at the moment.
    #def calculateInstantaneousNucleationRate(T: Union[float, Iterable[float]], action: Union[float, Iterable[float]], potential: AnalysablePotential):
    # Returns the integrand of N, where N is the number of nucleated bubbles.
    def calculateInstantaneousNucleationRate(self, T: float, action: float) -> float:
        HSq = self.calculateHubbleParameterSq(T)

        Gamma = self.calculateBubbleNucleationRate(T, action)

        return Gamma / (T*HSq**2)

    # TODO: We can optimise this for a list of input temperatures by reusing potential samples in adjacent derivatives.
    #  For now we don't handle a list of input temperatures so this doesn't matter.
    def calculateHubbleParameterSq(self, T: float) -> float:
        # Default is energy density for from phase.
        rhof = hydrodynamics.calculateEnergyDensityAtT_singlePhase(self.fromPhase, self.toPhase, self.potential, T)
        return 8*np.pi*GRAV_CONST/3*(rhof - self.groundStateEnergyDensity)

    def calculateHubbleParameterSq_fromHydro(self, hydroVars: HydroVars) -> float:
        return 8*np.pi*GRAV_CONST/3*hydroVars.energyDensityFalse

    def getHydroVars(self, T: float) -> HydroVars:
        return hydrodynamics.getHydroVars_new(self.fromPhase, self.toPhase, self.potential, T,
            self.groundStateEnergyDensity)

    def checkMilestoneTemperatures(self):
        TNew = self.actionSampler.subT[-1]
        TPrev = self.actionSampler.subT[-2]
        actionNew = self.actionSampler.subAction[-1]
        actionPrev = self.actionSampler.subAction[-2]

        # Unit nucleation (including phantom bubbles).
        if self.transition.Tn < 0 and self.numBubblesIntegral[-1] >= 1:
            interpFactor = (self.numBubblesIntegral[-1] - 1) / (self.numBubblesIntegral[-1] - self.numBubblesIntegral[-2])
            Tn = TPrev + interpFactor * (TNew - TPrev)
            Hn = self.hubbleParameter[-2] + interpFactor * (self.hubbleParameter[-1] - self.hubbleParameter[-2])
            self.transition.analysis.actionTn = actionPrev + (actionNew - actionPrev) \
                                                * (self.numBubblesIntegral[-1] - 1) / (
                                                            self.numBubblesIntegral[-1] - self.numBubblesIntegral[-2])
            self.transition.Tn = Tn
            self.transition.analysis.Hn = Hn
            self.transition.analysis.betaTn = Hn * Tn * (actionPrev - actionNew) / (TPrev - TNew)
            # Store the reheating temperature from this point, using conservation of energy.
            self.transition.Treh_n = self.calculateReheatTemperature(Tn)

        # Unit nucleation (excluding phantom bubbles).
        if self.transition.Tnbar < 0 and self.numBubblesCorrectedIntegral[-1] >= 1:
            interpFactor = (self.numBubblesCorrectedIntegral[-1] - 1) / (self.numBubblesCorrectedIntegral[-1] -
                                                                    self.numBubblesCorrectedIntegral[-2])
            Tnbar = TPrev + interpFactor * (TNew - TPrev)
            Hnbar = self.hubbleParameter[-2] + interpFactor * (self.hubbleParameter[-1] - self.hubbleParameter[-2])
            self.transition.analysis.actionTnbar = actionPrev + (actionNew - actionPrev) \
                                                   * (self.numBubblesCorrectedIntegral[-1] - 1) / (
                                                               self.numBubblesCorrectedIntegral[-1]
                                                               - self.numBubblesCorrectedIntegral[-2])
            self.transition.Tnbar = Tnbar
            self.transition.analysis.Hnbar = Hnbar
            self.transition.analysis.betaTnbar = Hnbar * Tnbar * (actionPrev - actionNew) / (TPrev - TNew)
            # Store the reheating temperature from this point, using conservation of energy.
            self.transition.Treh_nbar = self.calculateReheatTemperature(Tnbar)

        # Percolation.
        if self.transition.Tp < 0 and self.trueVacuumVolumeExtended[-1] >= self.percolationThreshold_Vext:
            # max(0, ...) for subcritical transitions, where it is possible that trueVacuumVolumeExtended[-2] > percThresh.
            interpFactor = max(0.0, (self.percolationThreshold_Vext - self.trueVacuumVolumeExtended[-2]) / (
                        self.trueVacuumVolumeExtended[-1] - self.trueVacuumVolumeExtended[-2]))
            Tp = TPrev + interpFactor * (TNew - TPrev)
            Hp = self.hubbleParameter[-2] + interpFactor * (self.hubbleParameter[-1] - self.hubbleParameter[-2])
            self.transition.analysis.actionTp = actionPrev + interpFactor * (actionNew - actionPrev)
            self.transition.Tp = Tp
            self.transition.analysis.Hp = Hp
            self.transition.analysis.betaTp = Hp * Tp * (actionPrev - actionNew) / (TPrev - TNew)

            # Also store whether the physical volume of the false vacuum was decreasing at Tp.
            # Make sure to cast to a bool, because JSON doesn't like encoding the numpy.bool type.
            self.transition.decreasingVphysAtTp = bool(self.physicalVolume[-1] < 0)

            # Store the reheating temperature from this point, using conservation of energy.
            self.transition.Treh_p = self.calculateReheatTemperature(Tp)

        # falseVacuumFraction = 1/e.
        if self.transition.Te < 0 and self.trueVacuumVolumeExtended[-1] >= 1:
            # max(0, ...) for subcritical transitions, where it is possible that trueVacuumVolumeExtended[-2] > 1.
            interpFactor = max(0.0, (1 - self.trueVacuumVolumeExtended[-2]) / (
                        self.trueVacuumVolumeExtended[-1] - self.trueVacuumVolumeExtended[-2]))
            Te = TPrev + interpFactor * (TNew - TPrev)
            He = self.hubbleParameter[-2] + interpFactor * (self.hubbleParameter[-1] - self.hubbleParameter[-2])
            self.transition.analysis.actionTe = actionPrev + interpFactor * (actionNew - actionPrev)
            self.transition.Te = Te
            self.transition.analysis.He = He
            self.transition.analysis.betaTe = He * Te * (actionPrev - actionNew) / (TPrev - TNew)

            # Store the reheating temperature from this point, using conservation of energy.
            self.transition.Treh_e = self.calculateReheatTemperature(Te)

        # Completion.
        if self.transition.Tf < 0 and self.falseVacuumFraction[-1] <= self.completionThreshold:
            if self.falseVacuumFraction[-1] == self.falseVacuumFraction[-2]:
                interpFactor = 0
            else:
                interpFactor = (self.falseVacuumFraction[-1] - self.completionThreshold) \
                               / (self.falseVacuumFraction[-1] - self.falseVacuumFraction[-2])
            Tf = TPrev + interpFactor * (TNew - TPrev)
            Hf = self.hubbleParameter[-2] + interpFactor * (self.hubbleParameter[-1] - self.hubbleParameter[-2])
            self.transition.analysis.actionTf = actionPrev + interpFactor * (actionNew - actionPrev)
            self.transition.Tf = Tf
            self.transition.analysis.Hf = Hf
            self.transition.analysis.betaTf = Hf * Tf * (actionPrev - actionNew) / (TPrev - TNew)

            # Also store whether the physical volume of the false vacuum was decreasing at Tf.
            # Make sure to cast to a bool, because JSON doesn't like encoding the numpy.bool type.
            self.transition.decreasingVphysAtTf = bool(self.physicalVolume[-1] < 0)

            # Store the reheating temperature from this point, using conservation of energy.
            self.transition.Treh_f = self.calculateReheatTemperature(Tf)

            if not self.bAnalyseTransitionPastCompletion:
                return True

        # Physical volume of the false vacuum is decreasing.
        if self.transition.TVphysDecr_high < 0 and self.physicalVolume[-1] < 0:
            if self.physicalVolume[-1] == self.physicalVolume[-2]:
                interpFactor = 0
            else:
                interpFactor = 1 - self.physicalVolume[-1] / (self.physicalVolume[-1] - self.physicalVolume[-2])
            self.transition.TVphysDecr_high = TPrev + interpFactor * (TNew - TPrev)

        # Physical volume of the false vacuum is increasing *again*.
        if self.transition.TVphysDecr_high > 0 and self.transition.TVphysDecr_low < 0 and self.physicalVolume[-1] > 0:
            if self.physicalVolume[-1] == self.physicalVolume[-2]:
                interpFactor = 0
            else:
                interpFactor = 1 - self.physicalVolume[-1] / (self.physicalVolume[-1] - self.physicalVolume[-2])
            self.transition.TVphysDecr_low = TPrev + interpFactor * (TNew - TPrev)

        return False

    # Whether we should stop integrating in transition analysis. Called by analyseTransition.
    def shouldTerminateTransitionAnalysis(self):
        return not self.bAnalyseTransitionPastCompletion and self.transition.Tf > 0

    def calculateReheatTemperature(self, T: float) -> float:
        Tsep = min(0.001*(self.transition.Tc - self.Tmin), 0.5*(T - self.Tmin))

        # Can't use supplied version of energy density calculation because T is changing. Default is from phase.
        rhof = hydrodynamics.calculateEnergyDensityAtT_singlePhase(self.fromPhase, self.toPhase, self.potential, T)

        def objective(t):
            rhot = hydrodynamics.calculateEnergyDensityAtT_singlePhase(self.fromPhase, self.toPhase, self.potential, t,
                forFromPhase=False)
            # Conservation of energy => rhof = rhof*Pf + rhot*Pt which is equivalent to rhof = rhot (evaluated at
            # different temperatures, T and Tt (Treh), respectively).
            return rhot - rhof

        # If the energy density of the true vacuum is never larger than the current energy density of the false vacuum
        # even at Tc, then reheating goes beyond Tc.
        if objective(self.transition.Tc) < 0:
            # Also, check the energy density of the true vacuum when it first appears. If a solution still doesn't exist
            # here, then just report -1.
            if self.toPhase.T[-1]-2*Tsep > self.transition.Tc and objective(self.toPhase.T[-1]-2*Tsep) < 0:
                return -1
            else:
                #return scipy.optimize.toms748(objective, T, self.toPhase.T[-1]-2*Tsep)
                # TODO: something broke when introducing Boltzmann suppression, can no longer use toms748 due to:
                #  zeros.py, line 959, in _compute_divided_differences
                #    row = np.diff(row)[:] / denom
                #  ValueError: operands could not be broadcast together with shapes (3,0) (2,)
                #  So, bisect instead...
                # We know that objective is negative at T and positive at sufficiently high temperature.
                # Pick the midpoint.
                low = T
                high = self.toPhase.T[-1]-2*Tsep
                mid = 0.5*(low + high)
                while (high - low) > 0.0001*self.potential.temperatureScale:
                    try:
                        result = objective(mid)
                    except:
                        traceback.print_exc()
                        print('Oops')
                    if result > 0:
                        high = mid
                    elif result < 0:
                        low = mid
                    else:
                        break
                    mid = 0.5*(low + high)
                return mid

        return scipy.optimize.toms748(objective, T, self.transition.Tc)

    def transitionCouldComplete(self, maxAction: float, Pf: list[float]) -> bool:
        if self.actionSampler.T[-1] <= self.potential.minimumTemperature:
            return False

        # If the nucleation rate is still high.
        if self.actionSampler.action[-1] < maxAction:
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


def calculateHubbleParameterSq(fromPhase: Phase, toPhase: Phase, potential: AnalysablePotential, T: float,
                               groundStateEnergyDensity: float) -> float:
    rhof = hydrodynamics.calculateEnergyDensityAtT_singlePhase(fromPhase, toPhase, potential, T, forFromPhase=True)
    return calculateHubbleParameterSq_supplied(rhof - groundStateEnergyDensity)


# TODO: We can optimise this for a list of input temperatures by reusing potential samples in adjacent derivatives.
#def calculateHubbleParameterSq_supplied(energyDensity: Union[list[float], np.ndarray])\
#        -> Union[list[float], np.ndarray]:
def calculateHubbleParameterSq_supplied(energyDensity: float) -> float:
    return 8*np.pi*GRAV_CONST/3*energyDensity


# TODO: This assumes vw = const and H(T) = H(1)*T**2. We can integrate to avoid these assumptions and hopefully better
#  handle supercooled phase transitions.
# TODO: [2023] just remove this?
def calculateTprime(T, R, vw, H):
    # Take H(T) = H(1)*T^2, vw(T) = const, a(T)/a(T') = T'/T.
    # R(T,T') = a(T) * int_T^T' vw(T'') / (T''*H(T'')*a(T'')) dT''
    #         = vw/T * int_T^T' 1 / H(T'') dT''
    #         = vw*T/H(T) * (1/T - 1/T')
    # T'(T,R) = T / (1 - R*H(T)/vw)
    return T / (1 - R*H/vw)


# TODO: currently unused. Is this necessary anymore?
# Note that the input H and Gamma are evaluated at the input T, i.e. H(T), Gamma(T).
def getEnergyForRadius(Tp, Tc, R, vw, H, GammaEff, T):
    # Find the nucleation time corresponding to the input radius R, such that a bubble that nucleated at time Tp would
    # have a radius of R at the current time T.
    Tprime = calculateTprime(Tp, R, vw, H)

    # If this bubble radius must have nucleated above the critical temperature, such bubbles cannot exist.
    if Tprime > Tc:
        return 0

    # If this bubble radius must have nucleated above the first temperature we sampled (which is above Tn), then we
    # assume the bubble has negligible chance of having nucleated and so doesn't contribute to the overall distribution.
    if Tprime > T[0]:
        return 0

    # Interpolation search:
    lowIndex = 0
    highIndex = len(T)-1
    index = 0

    while highIndex - lowIndex > 0:
        interpFactor = (T[lowIndex] - Tprime) / (T[lowIndex] - T[highIndex])
        index = int(round(lowIndex + interpFactor*(highIndex - lowIndex)))

        if T[index] == Tprime:
            break
        elif T[index] > Tprime:
            if lowIndex == index:
                break
            lowIndex = index
        else:
            if highIndex == index:
                break
            highIndex = index

    if T[index] > Tprime:
        interpFactor = (T[index] - Tprime) / (T[index] - T[index+1])
        Gammap = GammaEff[index] + interpFactor*(GammaEff[index+1] - GammaEff[index])
    else:
        interpFactor = (Tprime - T[index]) / (T[index-1] - T[index])
        Gammap = GammaEff[index] + interpFactor*(GammaEff[index-1] - GammaEff[index])

    # energy = R^3*dn/dR
    return R**3 * (Tp/Tprime)**4 * Gammap/vw


# TODO: currently unused. Need to verify correctness then use again.
def calculateTypicalLengthScale(Tp, indexTp, indexTn, Tc, meanBubbleSeparation, vw, Hin, rhoVin, GammaEffin, Tin,
        bPlot=False, bDebug=False):
    # Search for R that maximises the volume distribution: R^3 * dn/dR,
    #                         and the energy distribution: e(T, R) * dn/dR.

    # First, find the bubble radius at Tp for each temperature above Tp at which it could have nucleated.
    # That is, a bubble that nucleated at T' > Tp would have a radius of R(Tp, T') at Tp. We wish to find R.

    numSamples = 500

    if len(Tin) < numSamples:
        T = np.linspace(Tin[-1], Tin[0], numSamples)
        H = lagrange(Tin, Hin)(T)
        rhoV = lagrange(Tin, rhoVin)(T)
        GammaEff = lagrange(Tin, GammaEffin)(T)
    else:
        T = Tin
        H = Hin
        rhoV = rhoVin
        GammaEff = GammaEffin

    g = 100.0
    Mpl = 2.435e18
    xi = np.sqrt(np.pi**2*g/(90*Mpl**2))
    HradatTp = xi*Tp**2

    HatTp = H[indexTp]

    integralApprox = np.zeros(indexTp)
    #integralExact = np.zeros(indexTp)

    R = np.zeros(indexTp)
    Rapprox = np.zeros(indexTp)
    Rexact = np.zeros(indexTp)
    vol = np.zeros(indexTp)
    energy = np.zeros(indexTp)

    f = lambda x: vw/H[x]/T[x]

    for i in range(indexTp-2, -1, -1):
        # As temperature is increasing from Tp, the radius is increased according to the standard integral for R.
        R[i] = R[i+1] + 0.5*(f(i+1) + f(i))*(T[i] - T[i+1])
        Rapprox[i] = vw/HradatTp*(1 - Tp/T[i])
        vol[i] = 4*np.pi/3 * R[i]**3 * (Tp/T[i])**4 * GammaEff[i] / vw

        integralApprox[i] = (Tp - Tp**2/T[i]) / HatTp

    for i in range(indexTp-2, -1, -1):
        # Here T[i] = Tnuc.
        Rp = np.zeros(indexTp)

        # For a given R[i], we have T[i] = Tnuc(Tp, R[i]).
        # We still need R'(T', Tnuc) = R'(T[j], T[i])

        # calculate the R' array.
        for j in range(indexTp-2, max(0, i-1), -1):
            Rp[j] = T[j+1]/T[j]*Rp[j+1] + vw/T[j] * (1/H[j+1] + 1/H[j])*(T[j] - T[j+1])

        for j in range(indexTp-2, max(0, i-1), -1):
            energy[i] += 0.5*(T[j] - T[j+1]) * (1/T[j+1]*(vw/H[j+1] + Rp[j+1])*Rp[j+1]**2*rhoV[j+1]
                + 1/T[j]*(vw/H[j] + Rp[j])*Rp[j]**2*rhoV[j])

        energy[i] *= 4*np.pi/vw * GammaEff[i] * (Tp/T[i])**4

    TpApprox = np.zeros(len(R))

    for i in range(len(R)):
        TpApprox[i] = Tp/(1 - R[i]*HradatTp/vw)

    maxVolumeIndex = vol.argmax()
    maxEnergyIndex = energy.argmax()
    RVolMax = R[maxVolumeIndex]
    REnergyMax = R[maxEnergyIndex]

    REnergyScaled = R/REnergyMax

    if bDebug:
        print('Maximum energy radius:', REnergyMax, 'Energy:', energy.max(initial=0))
        print('Maximum volume radius:', RVolMax, 'Volume:', vol.max(initial=0))
        #print('Mean bubble separation:', meanBubbleSeparation, 'Energy:', energyFromAvgRad)
        print('Maximum energy nucleation temp:', T[maxEnergyIndex])
        print('Maximum volume nucleation temp:', T[maxVolumeIndex])
        #print('Mean separation nucleation temp:', calculateTprime(Tp, meanBubbleSeparation, vw, H))

    if bPlot:
        fig = plt.figure(figsize=(12, 8))
        plt.plot(R, T[:indexTp], lw=2.5)
        plt.plot(R, TpApprox, lw=2.5)
        plt.legend(['Numerical', 'Approx'])
        plt.xlabel('$R \,\, \\mathrm{[GeV^{-1}]}$', fontsize=24)
        plt.ylabel('$T\'(T_p,R) \,\, \\mathrm{[GeV]}$', fontsize=24)
        plt.tick_params(size=5, labelsize=16)
        plt.margins(0, 0)
        plt.show()

        plt.plot(T[:indexTp-1], R[:-1], c='b')
        plt.plot(T[:indexTp-1], Rapprox[:-1], c='r', ls='--')
        #plt.plot(T[:indexTp-1], np.log10(Rexact[:-1]), c='g', ls=':')
        plt.legend(['Numerical', 'Approx'])
        plt.xlabel('$T$')
        plt.ylabel('$R(T,T_{pw})$')
        plt.yscale('log')
        plt.show()

        # Don't specify 'initial' for the max functions here. These are likely to be well below 1 but we don't know how
        # much by, so there is no reasonable general choice for 'initial'.
        plt.plot(REnergyScaled, energy / energy.max())
        plt.plot(REnergyScaled, vol / vol.max())
        plt.axvline(meanBubbleSeparation / REnergyMax, ls='--')
        plt.xlabel('$R/R_{max}$')
        #plt.ylabel('$E(T,R)$')
        plt.legend(['Energy-weighted', 'Volume-weighted'])
        plt.show()

    return REnergyMax, RVolMax


# Returns T, action, bFinishedAnalysis.
def loadPrecomputedActionData(fileName: str, transition: Transition, maxActionThreshold: float) -> tuple[list[float],
        list[float], bool]:
    if fileName == '':
        return [], [], False

    precomputedT = []
    precomputedAction = []

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
            precomputedAction = transDict['action']

            # If the nucleation window wasn't found, we do not have precomputed data.
            if not transDict['foundNucleationWindow']:
                transition.analysis.T = precomputedT
                transition.analysis.action = precomputedAction
                return [], [], True
        else:
            print('Unable to find transition with id =', transition.ID, 'in the JSON file.')
    elif fileName[-4:] == '.txt':
        data = np.loadtxt(fileName)

        if data is not None:
            precomputedT = data[..., 0][::-1]
            precomputedAction = data[..., 1][::-1]
    else:
        print('Unsupported file extension', fileName.split('.')[-1], 'for precomputed action curve file.')

    if len(precomputedT) == 0:
        return [], [], False

    if len(precomputedAction) > 0:
        if min(precomputedAction) > maxActionThreshold:
            transition.analysis.T = precomputedT
            transition.analysis.action = precomputedAction
            transition.analysis.actionCurveFile = fileName
            return [], [], True

    return precomputedT, precomputedAction, False


if __name__ == "__main__":
    print('No script to run.')
