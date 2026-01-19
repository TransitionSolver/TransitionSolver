"""
Transition analysis
=====================
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Union

import numpy as np
import scipy.optimize
from scipy.interpolate import lagrange

from . import integration
from .phase_structure import Phase, Transition
from .. import action
from ..gws import hydrodynamics
from ..gws.hydrodynamics import HydroVars, hubble_squared_from_energy_density


logger = logging.getLogger(__name__)


class Timer:

    def __init__(self, limit, start_time=None):
        self.start_time = start_time if start_time else time.perf_counter()
        self.limit = limit

    def timeout(self):
        self.save()
        return self.limit > 0 and self.analysisElapsedTime > self.limit

    def save(self):
        self.analysisElapsedTime = time.perf_counter() - self.start_time


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
    # Whether a phase with a field value very close to zero compared to its characteristic value should be forced to
    # zero after optimisation, in preparation for action evaluation. This may reduce the effect of numerical errors
    # during optimisation. E.g. the field configuration for a phase might have phi1 = 1e-5, whereas it should be
    # identically zero. The phase is then shifted to phi1 = 0 before the action is evaluated. See
    # PhaseHistoryAnalysis.evaluate_action for details.
    bForcePhaseOnAxis: bool = False

    pd_settings = {'verbose': False, 'maxiter': 20, 'tunneling_findProfile_params': {
        'phitol': 1e-8, 'xtol': 1e-8}, 'V_spline_samples': 100}

    def __init__(self, transitionAnalyser: 'TransitionAnalyser', minSonTThreshold: float, maxSonTThreshold: float,
                 toleranceSonT: float, stepSizeMax=0.95, action_ct=True):  # TODO make false

        # Copy properties for concision.
        self.transitionAnalyser = transitionAnalyser
        self.bDebug = transitionAnalyser.bDebug
        self.potential = transitionAnalyser.potential
        self.fromPhase = transitionAnalyser.fromPhase
        self.toPhase = transitionAnalyser.toPhase
        self.Tmin = transitionAnalyser.Tmin
        self.Tmax = transitionAnalyser.Tmax
        self.action_ct = action_ct

        self.T = []
        self.SonT = []

        self.subT = []
        self.subSonT = []
        self.subRhof = []
        self.subRhot = []

        self.lower_s_on_t_data = []

        self.stepSize = -1
        self.stepSizeMax = stepSizeMax

        self.minSonTThreshold = minSonTThreshold
        self.maxSonTThreshold = maxSonTThreshold
        self.toleranceSonT = toleranceSonT

    def set_step_size(self, low=None, mid=None, high=None):
        lowT = self.T[-1] if low is None else low.T
        midT = self.T[-2] if mid is None else mid.T
        highT = self.T[-3] if high is None else high.T
        lowSonT = self.SonT[-1] if low is None else low.SonT
        midSonT = self.SonT[-2] if mid is None else mid.SonT
        highSonT = self.SonT[-3] if high is None else high.SonT

        # Check for linearity. Create a line between the lowest and highest temperature points, and check how far from
        # the average value the middle temperature point is.
        interpValue = lowSonT + (highSonT - lowSonT) * \
            (midT - lowT) / (highT - lowT)
        linearity = 1 - 2*abs(midSonT - interpValue) / abs(highSonT - lowSonT)

        if self.stepSize == -1:
            self.stepSize = min(self.stepSizeMax, 1 -
                                (abs(midSonT - lowSonT) / midSonT))

        # This gives (lin, stepFactor): (0.99, 1.0315), (0.98, 0.9186), (0.94, 0.8) and 0.5 for lin <= 0.8855.
        stepFactor = max(0.5, 0.8 + 0.4*((linearity - 0.94)/(1 - 0.94))**3)
        self.stepSize = min(self.stepSizeMax, 1 -
                            stepFactor*(1 - self.stepSize))

    def next_sampe(self, sampleData: ActionSample, Gamma: list[float], numBubbles: float) -> (bool, str):
        # If we are already near T=0, the transition is assumed to not progress from here. We consider only bubble
        # nucleation via thermal fluctuations.
        # if len(self.T) > 0 and sampleData.T <= 0.001:
        #    return False, 'Freeze out'

        # If we are already near the minimum temperature allowed for phase transitions in this potential, we assume the
        # transition will not progress from here. Or, if it does, we cannot accurately determine its progress due to
        # external effects (like other cosmological events) that we don't handle.
        if len(self.T) > 0 and sampleData.T <= self.potential.minimum_temperature:
            return False, 'Reached minimum temperature'

        # Remove all stored data points whose temperature is larger than the last sampled temperature.
        while len(self.lower_s_on_t_data) > 0 and self.lower_s_on_t_data[-1].T >= self.T[-1]:
            self.lower_s_on_t_data.pop()

        # Construct a quadratic Lagrange interpolant from the three most recent action samples.
        quadInterp = lagrange(self.T[-3:], self.SonT[-3:])
        # Extrapolate with the same step size as between the last two samples.
        Tnew = max(self.Tmin*1.001, 2*self.T[-1] - self.T[-2])
        SonTnew = quadInterp(Tnew)

        # If we are sampling the same point because we've reached Tmin, then the transition cannot progress any
        # further.
        if self.T[-1] == self.Tmin*1.001:
            logger.debug(
                'Already sampled near Tmin ={}. Transition analysis halted', sampleData.T)
            return False, 'Reached Tmin'

        # Determine the nucleation rate for nearby temperatures under the assumption that quadratic extrapolation is
        # appropriate.

        GammaNew = self.transitionAnalyser.gamma_rate(Tnew, SonTnew)
        GammaCur = self.transitionAnalyser.gamma_rate(
            self.T[-1], self.SonT[-1])
        GammaPrev = self.transitionAnalyser.gamma_rate(
            self.T[-2], self.SonT[-2])

        def nearMaxNucleation() -> bool:
            dSdTnew = (self.SonT[-1] - SonTnew)/(self.T[-1] - Tnew)
            dSdT = (self.SonT[-2] - self.SonT[-1])/(self.T[-2] - self.T[-1])
            # If the relative derivative is changing rapidly, then we are near the minimum.
            # TODO: given the exponential curve, might need to change this derivative test.
            derivTest = dSdTnew/dSdT < 0.8 if dSdT > 0 else dSdTnew/dSdT > 1.25
            return GammaNew > 0 and (GammaNew/GammaCur < 0.8*GammaCur/GammaPrev or derivTest)

        def step_factor():

            nearMaxFactor = 0.7

            if numBubbles < 1e-4:
                if GammaNew < GammaCur:
                    return 2.

                if nearMaxNucleation():
                    return nearMaxFactor

                extraBubbles = self.extra_bubbles(Tnew, SonTnew)

                if extraBubbles + numBubbles > 1e-4:
                    if extraBubbles > numBubbles*10:
                        if extraBubbles > numBubbles*100:
                            return 0.5
                        return 0.75
                    if extraBubbles < numBubbles:
                        return 2.
                if extraBubbles + numBubbles < 1e-5:
                    return 2.
                return 1.5

            if numBubbles < 0.1:
                if GammaNew < GammaCur:
                    return 1.4

                if nearMaxNucleation():
                    return nearMaxFactor

                extraBubbles = self.extra_bubbles(Tnew, SonTnew)

                if extraBubbles > numBubbles*5:
                    if extraBubbles > numBubbles*25:
                        return 0.5
                    return 0.75

                if extraBubbles < numBubbles:
                    return 1.4

            if numBubbles < 1:
                if GammaNew < GammaCur:
                    return 1.1

                if nearMaxNucleation():
                    return nearMaxFactor

                extraBubbles = self.extra_bubbles(Tnew, SonTnew)

                if extraBubbles > 0.2*numBubbles:
                    if extraBubbles > 0.5*numBubbles:
                        return 0.5
                    return 0.75

                if extraBubbles < 0.1*numBubbles:
                    return 1.2

            if GammaNew < GammaCur:
                if GammaCur < GammaPrev:
                    if GammaNew < 1e-3*max(Gamma):
                        return 1.5
                    return 1.2
                # Found minimum of Gamma, sample more densely.
                return 0.6

            if numBubbles < 100:

                if nearMaxNucleation():
                    return nearMaxFactor

                extraBubbles = self.extra_bubbles(Tnew, SonTnew)

                if numBubbles < 10:
                    if extraBubbles < numBubbles:
                        if extraBubbles < 0.5*numBubbles:
                            return 3.
                        return 2.
                    if extraBubbles < 2*numBubbles:
                        return 2.
                return 1.5
            return 1

        stepFactor = step_factor()

        if numBubbles < 1 and SonTnew < 160:
            if stepFactor > 1:
                stepFactor = 1 + 0.85 * (stepFactor - 1)
            else:
                stepFactor *= 0.85

        Tnew = max(self.Tmin*1.001, self.T[-1] - max(stepFactor*(self.T[-2] - self.T[-1]),
                                                     self.potential.minimum_temperature))

        # Prevent large steps near T=Tmin causing large errors in the interpolated action from affecting ultracooled
        # transitions.
        if (Tnew - self.Tmin) < 0.5*(self.T[-1] - self.Tmin) and (self.T[-1] - self.Tmin) > 5.:
            Tnew = 0.5*(self.Tmin + self.T[-1])

        # Hack for better resolution in supercool GW plots.
        # maxStep = 0.1 # BP1 and BP2
        # maxStep = 0.3 # BP3 and BP4

        # if abs(Tnew - self.T[-1]) > maxStep:
        #     Tnew = self.T[-1] - maxStep

        sampleData.T = Tnew

        sampleData.S3 = self.evaluate_action(sampleData.T)

        if not sampleData.is_valid:
            logger.info(
                'Failed to evaluate action at trial temperature T = {}', sampleData.T)
            return False, f'Action failed: T = {sampleData.T}. S3 = {sampleData.S3}'

        self.T.append(sampleData.T)
        self.SonT.append(sampleData.SonT)

        return True, 'Success'

    def extra_bubbles(self, Tnew: float, SonTnew: float) -> float:
        extraBubbles = 0
        numPoints = 20

        TList = np.linspace(self.T[-1], Tnew, numPoints)
        # TODO: replace with quadratic interpolation.
        SonTList = np.linspace(self.SonT[-1], SonTnew, numPoints)
        GammaList = self.transitionAnalyser.gamma_rate(TList, SonTList)
        energyStart = hydrodynamics.energy_density_from_phase(self.fromPhase, self.toPhase, self.potential,
                                                              TList[0]) - self.transitionAnalyser.groud_state_energy_density
        energyEnd = hydrodynamics.energy_density_from_phase(self.fromPhase, self.toPhase, self.potential,
                                                            TList[-1]) - self.transitionAnalyser.groud_state_energy_density
        # TODO: replace with quadratic interpolation.
        energyDensityList = np.linspace(energyStart, energyEnd, numPoints)
        HList = hubble_squared_from_energy_density(energyDensityList)
        integrandList = [GammaList[i]/(TList[i]*HList[i]**2)
                         for i in range(numPoints)]

        for i in range(1, numPoints):
            extraBubbles += 0.5 * \
                (integrandList[i] + integrandList[i-1]) * \
                (TList[i-1] - TList[i])

        return extraBubbles

    def num_sub_samples(self, minSonTThreshold, maxSonTThreshold):
        # Interpolate between the last two sample points, creating a densely sampled line which we can integrate.
        # Increase the sampling density as S/T decreases as we need greater resolution for S/T where the nucleation
        # probability rises exponentially.
        # TODO: make the number of interpolation samples have some physical motivation (e.g. ensure that the true
        # vacuum fraction can change by no more than 0.1% of its current value across each sample).
        quickFactor = 1.
        actionFactor = abs(self.SonT[-2]/self.SonT[-1] - 1) / \
            (1 - minSonTThreshold/maxSonTThreshold)
        dTFactor = abs(min(2, self.T[-2]/self.T[-1]) -
                       1) / (1 - self.Tmin/self.T[0])
        numSamples = max(
            1, int(quickFactor*(2 + np.sqrt(1000*(actionFactor + dTFactor)))))
        logger.debug('Num samples: {}, {},g {}',
                     numSamples, actionFactor, dTFactor)
        return numSamples

    def insert_lower_s_on_t_data(self, data):
        """
        Insert data into lower_s_on_t_data, maintaining the sorted order
        """
        idx = np.searchsorted([d.T for d in self.lower_s_on_t_data], data.T)
        self.lower_s_on_t_data.insert(idx, data)

    def evaluate_action(self, T) -> float:
        """
        @returns Action at specific temperature
        """
        from_field_config = self.fromPhase.find_phase_at_t(T, self.potential)
        to_field_config = self.toPhase.find_phase_at_t(T, self.potential)

        fieldSeparationScale = 0.001*self.potential.get_field_scale()

        if np.linalg.norm(from_field_config - to_field_config) < fieldSeparationScale:
            raise Exception(
                "The 'from' and 'to' phases merged after optimisation, in preparation for action evaluation.")

        if self.bForcePhaseOnAxis:
            for i in range(len(from_field_config.shape)):
                if abs(from_field_config[i]) < fieldSeparationScale:
                    from_field_config[i] = 0.0
                if abs(to_field_config[i]) < fieldSeparationScale:
                    to_field_config[i] = 0.0

        return self.evaluate_action_supplied(T, from_field_config, to_field_config)

    def evaluate_action_supplied(self, T, false_vacuum, true_vacuum) -> float:
        """
        @returns Action at specific temperature and between field configurations
        """
        if self.action_ct:
            return action.action_ct(self.potential, T, false_vacuum, true_vacuum, **self.pd_settings)
        return action.action_pt(self.potential, T, false_vacuum, true_vacuum, **self.pd_settings)


class ActionCurveShapeAnalysisData:
    def __init__(self):
        self.desiredData = None
        self.nextBelowDesiredData = None
        self.storedLowerActionData = []
        self.actionSamples = []

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

class LinearInterp:
    def __init__(self, x, target):
        self.factor = 0. if x[-1] == x[-2] else (x[-1] - target) / (x[-1] - x[-2])

    def __call__(self, z):
        return z[-2] + self.factor * (z[-1] - z[-2])


class TransitionAnalyser:
    bDebug: bool = False
    # Optimisation: check whether completion can occur before reaching T=0. If it cannot, stop the transition analysis.
    bCheckPossibleCompletion: bool = True
    # Whether transition analysis should continue after finding the completion temperature, all the way down to the
    # lowest temperature for which the transition is possible. If this is false, transition analysis stops as soon as
    # completion is found (default behaviour).
    bAnalyseTransitionPastCompletion: bool = False
    bAllowErrorsForTn: bool = True
    bReportAnalysis: bool = False
    timeout: float = -1.

    def __init__(self, potential, properties, fromPhase: Phase, toPhase: Phase,
                 groud_state_energy_density: float, Tmin=None, Tmax=None, vw=None, action_ct=True):  # TODO make false
        self.potential = potential
        self.properties = properties
        self.fromPhase = fromPhase
        self.toPhase = toPhase
        self.groud_state_energy_density = groud_state_energy_density
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.properties.use_cj_velocity = vw is None
        self.properties.vw = vw
        self.action_ct = action_ct

        if self.Tmin is None:
            # The minimum temperature for which both phases exist, and prevent analysis below the effective potential's
            # cutoff temperature. Below this cutoff temperature, external effects may dramatically affect the phase
            # transition and cannot be captured here in a generic way.
            self.Tmin = max(
                self.fromPhase.T[0], self.toPhase.T[0], self.potential.minimum_temperature)

        if self.Tmax is None:
            # The transition is not evaluated subcritically.
            self.Tmax = self.properties.Tc

        self.Tstep = max(
            0.0005*min(self.fromPhase.T[-1], self.toPhase.T[-1]), 0.0001*self.potential.get_temperature_scale())

    def init_properties(self, hydroVars, T, SonT):
        self.properties.totalNumBubbles = [0.]
        self.properties.totalNumBubblesCorrected = [0.]
        self.properties.Vext = [0.]
        self.properties.Pf = [1.]
        self.properties.bubbleNumberDensity = [0.]
        self.properties.meanBubbleRadiusArray = [0.]
        self.properties.betaArray = [None]
        self.properties.HArray = [hydroVars.hubble_constant]
        self.properties.vw_samples = [self.vw(hydroVars)]
        self.properties.gamma = [self.gamma_rate(T, SonT)]
        self.properties.physical_volume = [3]
        self.properties.numBubblesIntegrand = [self.num_bubbles_integrand(self.properties.gamma[0], self.actionSampler.T[0], self.properties.HArray[0])]
        self.properties.densityIntegrand = [self.density_integrand(self.properties.Pf[0], self.properties.gamma[0], self.actionSampler.T[0], self.properties.HArray[0])]

    def check_milestones(self):
        # Unit nucleation (including phantom bubbles)
        if self.properties.Tn is None and self.properties.totalNumBubbles[-1] >= 1:
            self.properties.idx_tn = len(self.properties.HArray) - 1
            li = LinearInterp(self.properties.totalNumBubbles, 1)
            self.properties.SonTn = li(self.actionSampler.subSonT)
            self.properties.Tn = li(self.actionSampler.subT)
            self.properties.Hn = li(self.properties.HArray)
            self.properties.betaTn = li(self.properties.betaArray)
            self.properties.Treh_n = self.reheat_temperature(self.properties.Tn)

        # Unit nucleation (excluding phantom bubbles)
        if self.properties.Tnbar is None and self.properties.totalNumBubblesCorrected[-1] >= 1:
            self.properties.idx_tnbar = len(self.properties.HArray) - 1
            li = LinearInterp(self.properties.totalNumBubblesCorrected, 1)
            self.properties.SonTnbar = li(self.actionSampler.subSonT)
            self.properties.Tnbar = li(self.actionSampler.subT)
            self.properties.Hnbar = li(self.properties.HArray)
            self.properties.betaTnbar = li(self.properties.betaArray)
            self.properties.Treh_nbar = self.reheat_temperature(self.properties.Tnbar)

        # Percolation
        if self.properties.Tp is None and self.properties.Vext[-1] >= self.percolationThreshold_Vext:
            self.properties.idx_tp = len(self.properties.HArray) - 1
            li = LinearInterp(self.properties.Vext,
                              self.percolationThreshold_Vext)
            self.properties.SonTp = li(self.actionSampler.subSonT)
            self.properties.Tp = li(self.actionSampler.subT)
            self.properties.Hp = li(self.properties.HArray)
            self.properties.betaTp = li(self.properties.betaArray)
            self.properties.decreasingVphysAtTp = self.properties.physical_volume[-1] < 0
            self.properties.Treh_p = self.reheat_temperature(self.properties.Tp)
            hydroVars = hydrodynamics.make_hydro_vars(
                # TODO why not pass vacuum energy
                self.fromPhase, self.toPhase, self.potential, self.properties.Tp)
            # TODO names should reflect that they are at TP
            self.properties.transitionStrength = hydroVars.alpha
            self.properties.meanBubbleRadius = li(self.properties.meanBubbleRadiusArray)
            self.properties.meanBubbleSeparation = li(self.properties.meanBubbleSeparationArray)

        # Pf = 1/e
        if self.properties.Te is None and self.properties.Vext[-1] >= 1:
            self.properties.idx_te = len(self.properties.HArray) - 1
            # max(0, ...) for subcritical transitions, where it is possible that self.properties.Vext[-2] > 1.
            li = LinearInterp(self.properties.Vext, 1)
            self.properties.SonTe = li(self.actionSampler.subSonT)
            self.properties.Te = li(self.actionSampler.subT)
            self.properties.He = li(self.properties.HArray)
            self.properties.betaTe = li(self.properties.betaArray)
            self.properties.Treh_e = self.reheat_temperature(self.properties.Te)

        # Completion
        if self.properties.Tf is None and self.properties.Pf[-1] <= self.completionThreshold:
            self.properties.idx_tf = len(self.properties.HArray) - 1
            li = LinearInterp(self.properties.Pf, self.completionThreshold)
            self.properties.SonTf = li(self.actionSampler.subSonT)
            self.properties.Tf = li(self.actionSampler.subT)
            self.properties.Hf = li(self.properties.HArray)
            self.properties.betaTf = li(self.properties.betaArray)
            self.properties.decreasingVphysAtTf = self.properties.physical_volume[-1] < 0
            self.properties.Treh_f = self.reheat_temperature(self.properties.Tf)

        # Physical volume of the false vacuum is decreasing
        if self.properties.TVphysDecr_high is None and self.properties.physical_volume[-1] < 0:
            self.properties.idx_tVphysDecr_high = len(
                self.properties.HArray) - 1
            li = LinearInterp(self.properties.physical_volume, 0)
            self.properties.TVphysDecr_high = li(self.actionSampler.subT)

        # Physical volume of the false vacuum is increasing *again*
        if self.properties.TVphysDecr_high is not None and self.properties.TVphysDecr_low is None and self.properties.physical_volume[-1] > 0:
            self.properties.idx_tVphysDecr_low = len(self.properties.HArray) - 1
            li = LinearInterp(self.properties.physical_volume, 0)
            self.properties.TVphysDecr_low = li(self.actionSampler.subT)

    def finalize_properties(self):
        self.properties.T = self.actionSampler.T
        self.properties.SonT = self.actionSampler.SonT
        self.properties.TSubSampleArray = self.actionSampler.subT
        self.properties.gamma_eff = [
            p * g for p, g in zip(self.properties.Pf, self.properties.gamma)]

    def update_properties(self, hydrovars, SonT, T, dT):
        self.properties.HArray.append(hydrovars.hubble_constant)
        self.properties.vw_samples.append(self.vw(hydrovars))
        self.properties.gamma.append(self.gamma_rate(T, SonT))
        self.properties.numBubblesIntegrand.append(self.num_bubbles_integrand(self.properties.gamma[-1], T, self.properties.HArray[-1]))
        self.properties.totalNumBubbles.append(self.properties.totalNumBubbles[-1] + 0.5 * dT * (self.properties.numBubblesIntegrand[-1] + self.properties.numBubblesIntegrand[-2]))
        # TODO: not a great derivative, can do better.
        self.properties.betaArray.append(self.properties.HArray[-1]*self.actionSampler.subT[-1]*(self.actionSampler.subSonT[-2] - self.actionSampler.subSonT[-1])/dT)

    def update_after_vac(self, data, T, dT, j):
        self.properties.Vext.append(4/3 * np.pi * data)
        self.properties.physical_volume.append(3 + T * (self.properties.Vext[-2] - self.properties.Vext[-1]) / dT)
        self.properties.Pf.append(np.exp(-self.properties.Vext[-1]))

        self.properties.densityIntegrand.append(self.density_integrand(self.properties.Pf[-1], self.properties.gamma[ j], T, self.properties.HArray[ j]))
        self.properties.bubbleNumberDensity.append((T / (T + dT))**3 * self.properties.bubbleNumberDensity[-1]
                               + 0.5*dT*T**3 * (self.properties.densityIntegrand[-2] + self.properties.densityIntegrand[-1]))

        self.properties.totalNumBubblesCorrected.append(self.properties.totalNumBubblesCorrected[-1] + 0.5 * dT * (self.properties.numBubblesIntegrand[-1] * self.properties.Pf[-1] + self.properties.numBubblesIntegrand[-2] * self.properties.Pf[-2]))

    def update_after_rad(self, data):
        self.properties.meanBubbleRadiusArray.append(data)

    def vw(self, hydrovars: HydroVars) -> float:
        """
        @returns Bubble wall velocity
        """
        if self.properties.use_cj_velocity:
            vw = hydrovars.cj_velocity
            if np.isnan(vw) or vw > 1.:
                logger.warning("vw = {}. Adjusting to 1", vw)
                return 1.
            return vw

        return self.properties.vw

    def density_integrand(self, Pf, gamma, T, H):
        return Pf * gamma / (T**4 * H)

    def num_bubbles_integrand(self, gamma, T, H):
        return gamma / (T * H**4)

    # TODO: need to handle subcritical transitions better. Shouldn't use integration if the max sampled action is well
    # below the nucleation threshold. Should treat the action as constant or linearise it and estimate transition
    # temperatures under that approximation.
    def analyseTransition(self, startTime: float = -1.0, percolationThreshold_Pf=0.71, completionThreshold=1e-2):

        self.percolationThreshold_Vext = -np.log(percolationThreshold_Pf)
        self.completionThreshold = completionThreshold # TODO mv to constructor

        # TODO: this should depend on the scale of the transition, so make it configurable.
        # Estimate the maximum significant value of S/T by finding where the instantaneous nucleation rate multiplied by
        # the maximum possible duration of the transition is O(1). This is highly conservative, but intentionally so
        # because we only sample maxSonTThreshold within some (loose) tolerance.
        maxSonTThreshold = self.estimateMaximumSignificantSonT() + 80
        minSonTThreshold = 80.0
        toleranceSonT = 3.0

        timer = Timer(self.timeout, startTime)

        self.actionSampler = ActionSampler(
            self, minSonTThreshold, maxSonTThreshold, toleranceSonT, action_ct=self.action_ct)

        sampleData = self.prime_transition_analysis(startTime)

        if timer.timeout():
            self.properties.error = "timed out"
            self.properties.analysed = False
            return

        if sampleData is None:
            self.properties.error = "no nucleation window"
            self.properties.analysed = False
            return

        # Remove any lower_s_on_t_data points that are very close together. We don't need to sample the S/T curve extremely
        # densely (a spacing of 1 is more than reasonable), and doing so causes problems with the subsequent steps along
        # the curve TODO: to be fixed anyway!
        if self.actionSampler.lower_s_on_t_data:
            keepIndices = [len(self.actionSampler.lower_s_on_t_data)-1]
            for i in range(len(self.actionSampler.lower_s_on_t_data)-2, -1, -1):
                # Don't discard the point if it is separated in temperature from the almost degenerate S/T value already
                # stored.
                if abs(self.actionSampler.lower_s_on_t_data[i].SonT -
                        self.actionSampler.lower_s_on_t_data[keepIndices[-1]].SonT) > 1 or\
                        abs(self.actionSampler.lower_s_on_t_data[i].T -
                            self.actionSampler.lower_s_on_t_data[keepIndices[-1]].T) >\
                        self.potential.get_temperature_scale()*0.001:
                    keepIndices.append(i)
                else:
                    logger.debug('Removing stored lower S/T data {} because it is too close to {}',
                                 self.actionSampler.lower_s_on_t_data[i], self.actionSampler.lower_s_on_t_data[keepIndices[-1]])

            self.actionSampler.lower_s_on_t_data = [
                self.actionSampler.lower_s_on_t_data[i] for i in keepIndices]

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

        

        # =================================================
        # Calculate the maximum temperature result.
        # This gives a boundary condition for integration.

        # The temperature for which S/T is minimised. This is important for determining the bubble number density. If this
        # minimum S/T value is encountered after the percolation temperature, then it can be ignored. It is only important
        # if it occurs before percolation.

        req_init = True

        def outerFunction_trueVacVol(x):
            return self.properties.gamma[x] / (self.actionSampler.subT[x]**4 * self.properties.HArray[x])

        def innerFunction_trueVacVol(x):
            return self.properties.vw_samples[x] / self.properties.HArray[x]

        def outerFunction_avgBubRad(x):
            return self.properties.gamma[x]*self.properties.Pf[x] / (self.actionSampler.subT[x]**4 * self.properties.HArray[x])

        def innerFunction_avgBubRad(x):
            return self.properties.vw_samples[x] / self.properties.HArray[x]

        def sampleTransformationFunction(x):
            return self.actionSampler.subT[x]

        # We need three (two) data points before we can initialise the integration helper for the true vacuum volume
        # (average bubble radius). We wait for four data points to be ready (see below for why), then initialise both at the
        # same time.
        integrationHelper_trueVacVol = None
        integrationHelper_avgBubRad = None

        # The index in the simulation we're up to analysing. Note that we always sample 1 past this index so we can
        # interpolate between T[simIndex] and T[simIndex+1].
        simIndex = 0

        # Keep sampling until we have identified the end of the phase transition or that the transition doesn't complete.
        # If bCheckPossibleCompletion is false we determine it does not complete only if we get to T=0 and it has not,
        # otherewise we can also determine this when could_complete returns false.
        while not self.bCheckPossibleCompletion or self.could_complete(maxSonTThreshold + toleranceSonT):
            # If the action begins increasing with decreasing temperature.
            if not self.properties.SonTmin and self.actionSampler.SonT[simIndex+1] > self.actionSampler.SonT[simIndex]:
                li = LinearInterp(self.actionSampler.SonT[:simIndex + 1], self.actionSampler.SonT[simIndex + 1])
                self.properties.Tmin = li(self.actionSampler.T)
                self.properties.SonTmin = li(self.actionSampler.SonT)

            numSamples = self.actionSampler.num_sub_samples(
                minSonTThreshold, maxSonTThreshold)
            T = np.linspace(
                self.actionSampler.T[simIndex], self.actionSampler.T[simIndex+1], numSamples)
            dT = T[0] - T[1]  # equally spaced so all same dT
            # We can only use quadratic interpolation if we have at least 3 action samples, which occurs for simIndex > 0.
            if simIndex > 0:
                quadInterp = lagrange(
                    self.actionSampler.T[simIndex-1:], self.actionSampler.SonT[simIndex-1:])
                SonT = quadInterp(T)
            else:
                SonT = np.linspace(
                    self.actionSampler.SonT[simIndex], self.actionSampler.SonT[simIndex+1], numSamples)

            T1 = self.actionSampler.T[simIndex]
            T2 = self.actionSampler.T[simIndex+1]
            hydroVars1 = self.get_hydro_vars(T1)
            hydroVars2 = self.get_hydro_vars(T2)
            hydroVarsInterp = [hydrodynamics.interpolate_hydro_vars(
                hydroVars1, hydroVars2, t) for t in T] # TODO expensive? why not just compute cf interpolate


            # ==========================================================================================================
            # Begin subsampling.
            # ==========================================================================================================

            # Don't handle the first element of the list, as it is either the boundary condition for the integration (if
            # this is the first integrated data point), or was handled as the last element of the previous data point.
            for i in range(len(T)):

                if i != 0 or req_init:
                  self.actionSampler.subT.append(T[i])
                  self.actionSampler.subSonT.append(SonT[i])
                  self.actionSampler.subRhof.append(
                      hydroVarsInterp[i].energyDensityFalse)
                  self.actionSampler.subRhot.append(
                      hydroVarsInterp[i].energyDensityTrue)
  
                if req_init:
                    self.init_properties(hydroVarsInterp[0], self.actionSampler.T[0], self.actionSampler.SonT[0])
                    req_init = False
                    continue
           
                self.update_properties(hydroVarsInterp[i], SonT[i], T[i], dT)  # for all i > 0

                # The integration helper needs three data points before it can be initialised. However, we wait for an
                # additional point so that we can immediately integrate and add it to the true vacuum fraction and false
                # vacuum probability arrays as usual below.
                if integrationHelper_trueVacVol is None and i == 3:
                    integrationHelper_trueVacVol = integration.CubedNestedIntegrationHelper([0, 1, 2],
                                                                                            outerFunction_trueVacVol, innerFunction_trueVacVol, sampleTransformationFunction)

                    for j, d in enumerate(integrationHelper_trueVacVol.data[1:]):
                        self.update_after_vac(d, T[j + 1], dT, j + 1)  # for i = 1, 2

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

                    # Don't add the first element since we have already stored self.properties.Vext[0] = 0, Pf[0] = 1, etc.
                    for d in integrationHelper_avgBubRad.data[1:]:
                        self.update_after_rad(d)  # for i = 1, 2

                if integrationHelper_trueVacVol is not None:
                    integrationHelper_trueVacVol.integrate(len(self.actionSampler.subT)-1)
                    self.update_after_vac(integrationHelper_trueVacVol.data[-1], T[-1], dT, -1) # for i >= 3
                    integrationHelper_avgBubRad.integrate(len(self.actionSampler.subT)-1)
                    self.update_after_rad(integrationHelper_avgBubRad.data[-1]) # for i >= 3

                # Check if we have reached any milestones (e.g. unit nucleation, percolation, etc.).

                self.check_milestones()

                if self.properties.completed and not self.bAnalyseTransitionPastCompletion:
                    break

            # ==========================================================================================================
            # End subsampling.
            # ==========================================================================================================

            if self.properties.completed and not self.bAnalyseTransitionPastCompletion:
                logger.debug('Found Tf, stopping sampling')
                break

            if sampleData.T <= self.Tmin:
                logger.debug(
                    'The transition does not complete before reaching Tmin')
                break

            # Choose the next value of S/T we're aiming to sample.
            success, message = self.actionSampler.next_sampe(
                sampleData, self.properties.gamma, self.properties.totalNumBubbles[-1])

            simIndex += 1

            if timer.timeout():
                self.properties.error = "timed out"
                self.properties.analysed = False
                return

            if not success:
                logger.debug(
                    'Terminating transition analysis after failing to get next action sample. Reason:', message)

                if message in ('Freeze out', 'Reached Tmin'):
                    break

                self.properties.analysed = False
                self.properties.error = f'failed at T = {sampleData.T}: {message}'
                return

        # ==============================================================================================================
        # End transition analysis.
        # ==============================================================================================================

        self.finalize_properties()


    def prime_transition_analysis(self, startTime: float):
        timer = Timer(self.timeout, startTime)
        TcData = ActionSample(self.properties.Tc)

        if self.bDebug:
            print(
                f'Bisecting to find S/T = {self.actionSampler.maxSonTThreshold} ± {self.actionSampler.toleranceSonT}')

        # Use bisection to find the temperature at which S/T ~ maxSonTThreshold.
        actionCurveShapeAnalysisData = self.findNucleationTemperatureWindow_refined()

        if timer.timeout():
            return

        # TODO: Maybe use the names from here anyway? The current set of names is a little confusing!
        data = actionCurveShapeAnalysisData.desiredData
        bisectMinData = actionCurveShapeAnalysisData.nextBelowDesiredData
        lower_s_on_t_data = actionCurveShapeAnalysisData.storedLowerActionData
        allSamples = actionCurveShapeAnalysisData.actionSamples

        # If we didn't find any action values near the nucleation threshold, we are done.
        if data is None or not data.is_valid:
            if len(allSamples) == 0:
                if self.bReportAnalysis:
                    print('No transition')
                    print('No action samples')
                return

            allSamples = np.array(allSamples)
            minSonTIndex = np.argmin(allSamples[:, 1])

            if self.bReportAnalysis:
                print('No transition')
                print('Lowest sampled S/T =',
                      allSamples[minSonTIndex, 1], 'at T =', allSamples[minSonTIndex, 0])

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

            self.properties.T = T
            self.properties.SonT = SonT
            return

        intermediateData = data.copy()

        self.actionSampler.lower_s_on_t_data = lower_s_on_t_data

        if self.bDebug:
            print(
                'Attempting to find next reasonable (T, S/T) sample below maxSonTThreshold...')

        if data.SonT > self.actionSampler.minSonTThreshold + 0.8 * (
                self.actionSampler.maxSonTThreshold - self.actionSampler.minSonTThreshold):
            if self.bDebug:
                print(
                    'Presumably not a subcritical transition curve, with current S/T near maxSonTThreshold.')
            subcritical = False
            # targetSonT is the first S/T value we would like to sample. Skipping this might lead to numerical errors in the
            # integration, and sampling at higher S/T values is numerically insignificant.
            # targetSonT = minSonTThreshold + 0.98*(min(maxSonTThreshold, data.SonT) - minSonTThreshold)
            targetSonT = data.SonT * self.actionSampler.stepSizeMax

            # Check if the bisection window can inform which temperature we should sample to reach the target S/T.
            if abs(bisectMinData.SonT - targetSonT) < 0.3 * (
                    self.actionSampler.maxSonTThreshold - self.actionSampler.minSonTThreshold):
                interpFactor = (targetSonT - bisectMinData.SonT) / \
                    (data.SonT - bisectMinData.SonT)
                intermediateData.T = bisectMinData.T + \
                    interpFactor * (data.T - bisectMinData.T)
            # Otherwise, check if the low S/T data can inform which temperature we should sample to reach the target S/T.
            elif len(lower_s_on_t_data) > 0 and abs(lower_s_on_t_data[-1].SonT - targetSonT) \
                    < 0.5 * (self.actionSampler.maxSonTThreshold - self.actionSampler.minSonTThreshold):
                interpFactor = (
                    targetSonT - lower_s_on_t_data[-1].SonT) / (data.SonT - lower_s_on_t_data[-1].SonT)
                intermediateData.T = lower_s_on_t_data[-1].T + \
                    interpFactor * (data.T - lower_s_on_t_data[-1].T)
            # Otherwise, all that's left to do is guess.
            else:
                # Try sampling S/T at a temperature just below where S/T = maxSonTThreshold, and determine where to sample
                # next based on the result.
                intermediateData.T = self.Tmin + 0.99*(data.T - self.Tmin)

            intermediateData.S3 = self.actionSampler.evaluate_action(
                intermediateData.T)

            if timer.timeout():
                return

            if not intermediateData.is_valid:
                self.properties.error = f'Failed to evaluate action at trial temperature T={intermediateData.T}'
                return

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
                    print('Trying again at T =', 0.5 *
                          (intermediateData.T + data.T))
                correctionSamplesTaken += 1
                # Step halfway across the interval and try again.
                intermediateData.T = 0.5*(intermediateData.T + data.T)
                intermediateData.S3 = self.actionSampler.evaluate_action(
                    intermediateData.T)

                if timer.timeout():
                    return

                # Store this point so we don't have to resample near it.
                self.actionSampler.insert_lower_s_on_t_data(
                    intermediateData.copy())

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
                return

            self.actionSampler.T.extend([data.T, intermediateData.T])
            self.actionSampler.SonT.extend([data.SonT, intermediateData.SonT])
        else:
            if self.bDebug:
                print(
                    'Presumably a subcritical transition curve, with current S/T significantly below maxSonTThreshold.')
            subcritical = True

            # Take a very small step, with the size decreasing as S/T decreases.
            interpFactor = 0.99 + 0.009 * \
                (1 - data.SonT/self.actionSampler.minSonTThreshold)
            intermediateData.T = self.Tmin + interpFactor*(data.T - self.Tmin)

            # There's no point taking a tiny temperature step if we already took a larger step away from Tmax as our highest
            # sample point.
            intermediateData.T = min(intermediateData.T, 2*data.T - self.Tmax)
            intermediateData.S3 = self.actionSampler.evaluate_action(
                intermediateData.T)

            if timer.timeout():
                return

            # Don't accept cases where S/T is negative (translated to 0) for the last two evaluations. This suggests the
            # bounce solver is failing and we cannot proceed reliably.
            if not intermediateData.is_valid or intermediateData.SonT == 0 and TcData.SonT == 0:
                self.properties.error = f'Failed to evaluate action at trial temperature T={intermediateData.T}'
                # print('This was for a subcritical transition with initial S/T:', data.SonT, 'at T:', data.T)
                return

            # If we couldn't sample all the way to Tmax, predict what the action would be at Tmax and store that as a
            # previous sample to be used in the following integration. intermediateData will be used as the next sample
            # point.
            if data.T < self.Tmax:
                maxSonT = data.SonT + (data.SonT - intermediateData.SonT) * (self.Tmax - data.T) / (
                    data.T - intermediateData.T)

                self.actionSampler.T.extend([self.Tmax, data.T])
                self.actionSampler.SonT.extend([maxSonT, data.SonT])

                # We have already sampled this data point and should use it as the next point in the integration. Storing it
                # in lower_s_on_t_data automatically results in this desired behaviour. No copy is required as we don't alter
                # intermediateData from this point on.
                self.actionSampler.insert_lower_s_on_t_data(intermediateData)
            # If we sampled all the way to Tmax, use the sample there and intermediateData as the samples for the following
            # integration.
            else:
                self.actionSampler.T.extend([data.T, intermediateData.T])
                self.actionSampler.SonT.extend(
                    [data.SonT, intermediateData.SonT])

        if self.bDebug:
            print('Found next sample: T =', intermediateData.T,
                  'and S/T =', intermediateData.SonT)

        # Now take the same step in temperature and evaluate the action again.
        sampleData = intermediateData.copy()
        sampleData.T = 2 * intermediateData.T - data.T

        if not subcritical and len(self.actionSampler.lower_s_on_t_data) > 0 and self.actionSampler.lower_s_on_t_data[-1].T >= sampleData.T:
            if self.actionSampler.lower_s_on_t_data[-1].T < self.actionSampler.T[-1]:
                sampleData = self.actionSampler.lower_s_on_t_data[-1].copy()
            else:
                sampleData.S3 = self.actionSampler.evaluate_action(
                    sampleData.T)

                if timer.timeout():
                    return

            self.actionSampler.lower_s_on_t_data.pop()
        else:
            sampleData.S3 = self.actionSampler.evaluate_action(sampleData.T)

            if timer.timeout():
                return

            if not sampleData.is_valid:
                print('Failed to evaluate action at trial temperature T=', sampleData.T)

        self.actionSampler.set_step_size(
            low=sampleData, mid=intermediateData, high=data)

        # We have already sampled this data point and should use it as the next point in the integration. Storing it in
        # lower_s_on_t_data automatically results in this desired behaviour. For a near-instantaneous subcritical transition,
        # this will actually be the *second* next point in the integration, as the handling of intermediateData is also
        # postponed, and should be done before sampleData.
        self.actionSampler.insert_lower_s_on_t_data(sampleData.copy())

        return sampleData

    def estimateMaximumSignificantSonT(self, tolerance=2.0):
        actionMin = 50
        actionMax = 200
        action = 160

        while actionMax - actionMin > tolerance:
            action = 0.5*(actionMin + actionMax)

            nucRate = self.instantaneous_nucleation_rate(self.Tmax, action)
            numBubbles = nucRate*(self.Tmax - self.Tmin)

            if 0.1 < numBubbles < 10:
                return action

            if numBubbles < 1:
                actionMax = action
            else:
                actionMin = action

        return action

    def sign_label(self, data):
        if not data.is_valid:
            return 'unknown'

        if abs(data.SonT - self.actionSampler.maxSonTThreshold) <= self.actionSampler.toleranceSonT:
            return 'equal'

        if data.SonT > self.actionSampler.maxSonTThreshold:
            return 'above'

        return 'below'

    def findNucleationTemperatureWindow_refined(self):
        actionCurveShapeAnalysisData = ActionCurveShapeAnalysisData()
        Tstep = 0.01*(self.Tmax - self.Tmin)
        actionSamples = []
        lowerActionData = []

        TLow = self.Tmin
        THigh = self.Tmax

        def record_action(data):
            data.S3 = self.actionSampler.evaluate_action(data.T)
            actionSamples.append((data.T, data.SonT, data.is_valid))
            if data.is_valid and self.actionSampler.minSonTThreshold < data.SonT < self.actionSampler.maxSonTThreshold:
                lowerActionData.append(data.copy())

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
                actionCurveShapeAnalysisData.copyNextBelowDesiredData(
                    lowerTSampleToUse)
                # We probably just stored data in lowerActionData, so remove it since we're using it as the target value.
                if self.actionSampler.minSonTThreshold < dataToUse.SonT < self.actionSampler.maxSonTThreshold:
                    lowerActionData.pop()
            actionCurveShapeAnalysisData.copyActionSamples(actionSamples)
            actionCurveShapeAnalysisData.copyStoredLowerActionData(
                lowerActionData)

        # Evaluate the action at Tmin. (Actually just above so we avoid issues where the phase might disappear.)
        data = ActionSample(self.Tmin+Tstep)
        record_action(data)
        labelLow = self.sign_label(data)

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
            record_action(data)

            if data.is_valid:
                labelLow = self.sign_label(data)
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

            record_action(data)

            labelHigh = self.sign_label(data)

            seenBelowAtTHigh = labelHigh == 'below'

            # If the action at the high temperature is too low, sample closer to Tmax.
            while labelHigh == 'below' and abs(self.Tmax - data.T)/self.Tmax > 1e-4:
                lowerTSample = data.copy()
                data.T = 0.5*(data.T + self.Tmax)
                Tstep = 0.1*(self.Tmax - data.T)
                record_action(data)

                # TODO: cleanup if this still works [2023]
                # calculateActionSimple(potential, data, fromPhase, toPhase, Tstep=-Tstep, bDebug=settings.bDebug)
                # actionSamples.append((data.T, data.SonT, data.is_valid))
                labelLow = 'below'
                labelHigh = self.sign_label(data)

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
            record_action(data)

            Tstep *= 0.5
            labelMid = self.sign_label(data)

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
            record_action(data)

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
                TPredicted = midT - \
                    (midSonT - self.actionSampler.maxSonTThreshold)*dTdS
                extrapolationFailed = TPredicted < TLow
            else:
                TPredicted = data.T - \
                    (data.SonT - self.actionSampler.maxSonTThreshold)*dTdS
                extrapolationFailed = TPredicted > THigh

            if extrapolationFailed:
                if self.bDebug:
                    print('S/T will not go below', self.actionSampler.maxSonTThreshold,
                          'based on linear extrapolation.')

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
            record_action(data)

            prevLabelMid = labelMid
            labelMid = self.sign_label(data)

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

    def gamma_rate(self, T: Union[float, np.ndarray], action: Union[float, np.ndarray])\
            -> Union[float, np.ndarray]:
        return T**4 * (action/(2*np.pi))**(3/2) * np.exp(-action)

    def instantaneous_nucleation_rate(self, T: float, action: float) -> float:
        HSq = self.hubble_squared(T)
        Gamma = self.gamma_rate(T, action)
        return Gamma / (T * HSq**2)

    # TODO: We can optimise this for a list of input temperatures by reusing potential samples in adjacent derivatives.
    def hubble_squared(self, T: float) -> float:
        rhof = hydrodynamics.energy_density_from_phase(
            self.fromPhase, self.toPhase, self.potential, T)
        return hubble_squared_from_energy_density(rhof - self.groud_state_energy_density)

    def get_hydro_vars(self, T: float) -> HydroVars:
        return hydrodynamics.make_hydro_vars(self.fromPhase, self.toPhase, self.potential, T,
                                             self.groud_state_energy_density)

    def reheat_temperature(self, T: float) -> float:
        Tsep = min(0.001*(self.properties.Tc - self.Tmin), 0.5*(T - self.Tmin))
        rhof = hydrodynamics.energy_density_from_phase(
            self.fromPhase, self.toPhase, self.potential, T)

        def objective(t):
            rhot = hydrodynamics.energy_density_to_phase(
                self.fromPhase, self.toPhase, self.potential, t)
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
            # here, return
            if max_t > self.properties.Tc and objective(max_t) < 0:
                warnings.warn("Cannot find reheat temperature")
                return

        return scipy.optimize.toms748(objective, T, max_t)

    def could_complete(self, maxAction: float) -> bool:
        if self.actionSampler.T[-1] <= self.potential.minimum_temperature:
            return False

        # If the nucleation rate is still high.
        if self.actionSampler.SonT[-1] < maxAction:
            return True

        # Check if the transition progress is speeding up (i.e. the rate of change of Pf is increasing).
        if self.properties.Pf[-2] - self.properties.Pf[-1] > self.properties.Pf[-3] - self.properties.Pf[-2]:
            return True

        # If the transition has stagnated (or hasn't even begun).
        if self.properties.Pf[-1] == self.properties.Pf[-2]:
            return False

        # Assume the transition progress (i.e. change in Pf) is linear from here. Extrapolate to Tmin. Predict what
        # temperature would yield P(T) = 0.
        T0 = self.actionSampler.subT[-1] - self.properties.Pf[-1] * (self.actionSampler.subT[-2] - self.actionSampler.subT[-1])\
            / (self.properties.Pf[-2] - self.properties.Pf[-1])

        # If this temperature is above the minimum temperature for which this transition can still occur, then it's possible
        # that it could complete.
        return T0 >= self.Tmin


def hubble_squared(fromPhase: Phase, toPhase: Phase, potential, T: float,
                   groud_state_energy_density: float) -> float:
    rhof = hydrodynamics.energy_density_from_phase(
        fromPhase, toPhase, potential, T)
    return hubble_squared_from_energy_density(rhof - groud_state_energy_density)
