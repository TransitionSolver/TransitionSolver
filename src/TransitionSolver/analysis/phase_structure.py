"""
Represent phase structure
=========================
"""

import warnings

import numpy as np
from scipy.optimize import fmin_powell


class Phase:
    """
    Represent a phase from PT data
    """
    def __init__(self, key: float, phase: np.ndarray):
        self.key = int(key)
        self.T = phase[:, 0].T
        self.V = phase[:, 1].T
        self.phi = phase[:, 2:].T
        self.raw = phase

    def find_phase_at_t(self, T: float, potential, rel_tol=10) -> np.ndarray:
        """
        @returns Location of the phase at the given temperature by using interpolation between the stored data points,
        and local optimisation from that interpolated point
        """
        if T < self.T[0] or T > self.T[-1]:
            raise ValueError(f'Attempted to find phase {self.key} at T={T}, while defined only for'
                f'[{self.T[0]}, {self.T[-1]}]')

        idx = np.absolute(self.T - T).argmin()
        
        if self.T[idx] == T:
            return self.phi[..., idx]

        nxt_idx = idx - 1 if self.T[idx] > T else idx + 1
        dphi = self.phi[..., idx] - self.phi[..., nxt_idx]
        dt = self.T[idx] - self.T[nxt_idx]
        linear_interp = self.phi[..., idx] + (T - self.T[idx]) * dphi / dt
        direc = 0.5 * np.diag(dphi)
        
        optimised_point = fmin_powell(lambda X: potential(X, T), linear_interp, disp=False, direc=direc)

        if np.linalg.norm(optimised_point - linear_interp) > rel_tol * np.linalg.norm(dphi):
            warnings.warn("Using linear interpolation as the optimiser probably found a different phase")
            return linear_interp

        return np.atleast_1d(optimised_point)


class Transition:
    """
    Represent a transition from PT data and store data about that
    transition
    """
    def __init__(self, transition: np.ndarray):
        self.raw = transition
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

    @property
    def completed(self):
        return self.Tf >= 0.

    def report(self, reportFileName) -> dict:
        """
        @returns Data about transition collated into a dictionary
        """
        report = {}

        report['id'] = self.ID
        report['falsePhase'] = self.false_phase
        report['truePhase'] = self.true_phase
        report['analysed'] = self.analysis is not None
        report['subcritical'] = self.subcritical

        if self.analysis is not None:
            report['completed'] = self.completed
            if len(self.analysis.error) > 0: report['error'] = self.analysis.error
            # If we are updating the report file and reused its action data (i.e. if actionCurveFile == reportFileName),
            # make sure to keep that data in the report rather than create a circular reference.
            if self.analysis.actionCurveFile == '' or self.analysis.actionCurveFile == reportFileName:
                report['T'] = self.analysis.T
                report['SonT'] = self.analysis.SonT
            else:
                report['actionCurveFile'] = self.analysis.actionCurveFile

            report['vw'] = self.vw
            report['N'] = self.totalNumBubbles
            report['Nbar'] = self.totalNumBubblesCorrected

        report['Tc'] = self.Tc
        if self.Tn > 0:
            report['Tn'] = self.Tn
            report['SonTn'] = self.analysis.SonTn
        if self.Tnbar > 0:
            report['Tnbar'] = self.Tnbar
            report['SonTnbar'] = self.analysis.SonTnbar

        
        if self.Tp > 0: report['Tp'] = self.Tp
        if self.Te > 0: report['Te'] = self.Te
        if self.Tf > 0: report['Tf'] = self.Tf

        if self.TVphysDecr_high > 0: report['TVphysDecr_high'] = self.TVphysDecr_high
        if self.TVphysDecr_low > 0: report['TVphysDecr_low'] = self.TVphysDecr_low

        if self.Treh_n > 0: report['Treh_n'] = self.Treh_n
        if self.Treh_nbar > 0: report['Treh_nbar'] = self.Treh_nbar
        if self.Treh_p > 0: report['Treh_p'] = self.Treh_p
        if self.Treh_e > 0: report['Treh_e'] = self.Treh_e
        if self.Treh_f > 0: report['Treh_f'] = self.Treh_f

        if self.Tn > 0: report['betaTn'] = self.analysis.betaTn
        if self.Tnbar > 0: report['betaTnbar'] = self.analysis.betaTnbar
        if self.Tp > 0: report['betaTp'] = self.analysis.betaTp
        if self.Te > 0: report['betaTe'] = self.analysis.betaTe
        if self.Tf > 0: report['betaTf'] = self.analysis.betaTf
        if self.TGammaMax > 0: report['betaV'] = self.analysis.betaV

        if self.Tn > 0: report['Hn'] = self.analysis.Hn
        if self.Tnbar > 0: report['Hnbar'] = self.analysis.Hnbar
        if self.Tp > 0: report['Hp'] = self.analysis.Hp
        if self.Te > 0: report['He'] = self.analysis.He
        if self.Tf > 0: report['Hf'] = self.analysis.Hf

        if self.Tmin > 0:
            report['Tmin'] = self.Tmin
            report['SonTmin'] = self.SonTmin
        if self.TGammaMax > 0:
            report['TGammaMax'] = self.TGammaMax
            report['SonTGammaMax'] = self.SonTGammaMax
            report['GammaMax'] = self.GammaMax
        #if self.Teq > 0:
        report['Teq'] = self.Teq
        #report['SonTeq'] = self.SonTeq

        if self.Tp > 0: report['decreasingVphysAtTp'] = self.decreasingVphysAtTp
        if self.Tf > 0: report['decreasingVphysAtTf'] = self.decreasingVphysAtTf

        report['foundNucleationWindow'] = self.bFoundNucleationWindow

        if self.Tp > 0:
            report['meanBubbleSeparation'] = self.meanBubbleSeparation
            report['meanBubbleRadius'] = self.meanBubbleRadius
            report['energyWeightedBubbleRadius'] = self.energyWeightedBubbleRadius
            report['volumeWeightedBubbleRadius'] = self.volumeWeightedBubbleRadius
            report['transitionStrength'] = self.transitionStrength

        if len(self.TSubampleArray) > 0: report['TSubsample'] = self.TSubampleArray
        if len(self.betaArray) > 0: report['beta'] = self.betaArray
        if len(self.HArray) > 0: report['H'] = self.HArray
        if len(self.meanBubbleSeparationArray) > 0: report['meanBubbleSeparationArray'] = self.meanBubbleSeparationArray
        if len(self.meanBubbleRadiusArray) > 0: report['meanBubbleRadiusArray'] = self.meanBubbleRadiusArray
        if len(self.Pf) > 0: report['Pf'] = self.Pf

        return report


class PhaseStructure:

    def __init__(self, phases=None, transitions=None, paths=None):
        self.phases = [] if not phases else phases
        self.transitions = [] if not transitions else transitions
        self.paths = [] if not paths else paths
        self.transitions.sort(key=lambda x: x.ID)

    @property
    def groud_state_energy_density(self):
        """        
        @returns The lowest energy of any phase at T = 0
        """
        groud_state_energy_density = np.inf

        for phase in self.phases:
            if phase.T[0] == 0 and phase.V[0] < groud_state_energy_density:
                groud_state_energy_density = phase.V[0]

        if np.isinf(groud_state_energy_density):
            warnings.warn("Could not determine ground state energy density; assuming 0")
            groud_state_energy_density = 0.

        return groud_state_energy_density
