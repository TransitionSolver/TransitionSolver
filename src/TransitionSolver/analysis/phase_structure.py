"""
Analyse phase structure
=======================
"""

from typing import Union

import numpy as np
from scipy import optimize


class Phase:

    def __init__(self, key: float, phase: np.ndarray):
        self.key = int(key)
        self.T = phase[:, 0].T
        self.V = phase[:, 1].T
        self.phi = phase[:, 2:].T

    # Returns the location of the phase at the given temperature by using interpolation between the stored data points,
    # and local optimisation from that interpolated point.
    def findPhaseAtT(self, T: float, potential) -> np.ndarray:
        if T < self.T[0] or T > self.T[-1]:
            raise ValueError(f'Attempted to find phase {self.key} at T={T}, while defined only for'
                f'[{self.T[0]}, {self.T[-1]}]')

        minIndex = 0
        maxIndex = len(self.T)

        # Binary search to find the boundary temperature data (specifically, the indices in the T array).
        while maxIndex - minIndex > 1:
            midpoint = (minIndex + maxIndex) // 2

            if self.T[midpoint] == T:
                return self.phi[..., midpoint]
            elif self.T[midpoint] < T:
                minIndex = midpoint
            else:
                maxIndex = midpoint

        def V(X) -> Union[float, np.ndarray]: return potential(X, T)
        offset = 0.001*potential.get_field_scale()

        # Interpolate between the most relevant data points.
        if minIndex == maxIndex:
            interpPoint = self.phi[..., minIndex]
        else:
            interpFactor = (T - self.T[minIndex]) / (self.T[maxIndex] - self.T[minIndex])
            interpPoint = self.phi[..., minIndex] + interpFactor*(self.phi[..., maxIndex] - self.phi[..., minIndex])

        direction = np.diag(np.ones(potential.get_n_scalars())*offset)
        optimisedPoint = optimize.fmin_powell(V, interpPoint, disp=False, direc=direction)

        if len(optimisedPoint.shape) == 0:
            optimisedPoint = optimisedPoint.ravel()

        if abs(T - self.T[minIndex]) < 0.2*potential.get_temperature_scale() and \
                np.linalg.norm(optimisedPoint - interpPoint) > 0.2*potential.get_field_scale():
            # The minimum shifted too far, so the optimiser probably found a different phase.
            # Return the previous point.
            return interpPoint

        return optimisedPoint


class Transition:

    def __init__(self, transition: np.ndarray):
        self.Tc = transition[2]
        self.key = transition[-3]
        # The ID is used to match up the indices in a transition path (which is a list of transition ids).
        self.ID = int(transition[-2])
        self.subcritical = transition[-1] > 0
        self.false_phase = int(transition[0])
        self.true_phase = int(transition[1])

        self.Tn = -1.
        self.Tnbar = -1.
        self.Tp = -1.
        self.Te = -1.
        self.Tf = -1.
        self.TVphysDecr_high = -1.
        self.TVphysDecr_low = -1.
        self.Treh_n = -1
        self.Treh_nbar = -1
        self.Treh_p = -1
        self.Treh_e = -1
        self.Treh_f = -1

        self.Tmin = -1.
        self.TGammaMax = -1.
        self.Teq = -1.
        self.SonTmin = -1.
        self.SonTGammaMax = -1.

        self.Gamma = -1.

        self.decreasingVphysAtTp = False
        self.decreasingVphysAtTf = False

        self.vw = -1.
        self.analysis = None

        self.meanBubbleRadius = -1.
        self.meanBubbleSeparation = -1.
        self.energyWeightedBubbleRadius = -1.
        self.volumeWeightedBubbleRadius = -1.

        self.transitionStrength = -1.

        self.totalNumBubbles = 0.
        self.totalNumBubblesCorrected = 0.

        self.bFoundNucleationWindow = False

        self.TSubampleArray = []
        self.betaArray = []
        self.meanBubbleSeparationArray = []
        self.meanBubbleRadiusArray = []
        self.HArray = []
        self.Pf = []

    def report(self) -> dict:
        report = {}

        report['id'] = self.ID
        report['falsePhase'] = self.false_phase
        report['truePhase'] = self.true_phase
        report['analysed'] = self.analysis is not None
        report['subcritical'] = bool(self.subcritical)

        if self.analysis is None:
            return report

        report['completed'] = bool(self.Tf >= 0.0)
        report['error'] = self.analysis.error
        report['T'] = self.analysis.T
        report['SonT'] = self.analysis.SonT
        report['vw'] = self.vw
        report['N'] = self.totalNumBubbles
        report['Nbar'] = self.totalNumBubblesCorrected
        report['Tc'] = self.Tc
        report['Tn'] = self.Tn
        report['SonTn'] = self.analysis.SonTn
        report['Tnbar'] = self.Tnbar
        report['SonTnbar'] = self.analysis.SonTnbar
        report['Tp'] = self.Tp
        report['Te'] = self.Te
        report['Tf'] = self.Tf
        report['TVphysDecr_high'] = self.TVphysDecr_high
        report['TVphysDecr_low'] = self.TVphysDecr_low
        report['Treh_n'] = self.Treh_n
        report['Treh_nbar'] = self.Treh_nbar
        report['Treh_p'] = self.Treh_p
        report['Treh_e'] = self.Treh_e
        report['Treh_f'] = self.Treh_f
        report['betaTn'] = self.analysis.betaTn
        report['betaTnbar'] = self.analysis.betaTnbar
        report['betaTp'] = self.analysis.betaTp
        report['betaTe'] = self.analysis.betaTe
        report['betaTf'] = self.analysis.betaTf
        report['betaV'] = self.analysis.betaV
        report['Hn'] = self.analysis.Hn
        report['Hnbar'] = self.analysis.Hnbar
        report['Hp'] = self.analysis.Hp
        report['He'] = self.analysis.He
        report['Hf'] = self.analysis.Hf
        report['Tmin'] = self.Tmin
        report['SonTmin'] = self.SonTmin
        report['TGammaMax'] = self.TGammaMax
        report['SonTGammaMax'] = self.SonTGammaMax
        report['GammaMax'] = self.GammaMax
        report['Teq'] = self.Teq
        report['decreasingVphysAtTp'] = self.decreasingVphysAtTp
        report['decreasingVphysAtTf'] = self.decreasingVphysAtTf
        report['foundNucleationWindow'] = self.bFoundNucleationWindow
        report['meanBubbleSeparation'] = self.meanBubbleSeparation
        report['meanBubbleRadius'] = self.meanBubbleRadius
        report['energyWeightedBubbleRadius'] = self.energyWeightedBubbleRadius
        report['volumeWeightedBubbleRadius'] = self.volumeWeightedBubbleRadius
        report['transitionStrength'] = self.transitionStrength
        report['TSubsample'] = self.TSubampleArray
        report['beta'] = self.betaArray
        report['H'] = self.HArray
        report['meanBubbleSeparationArray'] = self.meanBubbleSeparationArray
        report['meanBubbleRadiusArray'] = self.meanBubbleRadiusArray
        report['Pf'] = self.Pf

        return report


class PhaseStructure:

    def __init__(self, phases=None, transitions=None, paths=None):
        self.phases = [] if not phases else phases
        self.transitions = [] if not transitions else transitions
        self.transitionPaths = [] if not paths else paths

    @property
    def ground_state_energy(self):
        ground_state_energy = np.inf

        for phase in self.phases:
            if phase.T[0] == 0 and phase.V[0] < ground_state_energy:
                ground_state_energy = phase.V[0]

        if ground_state_energy == np.inf:
            return 0.

        return ground_state_energy